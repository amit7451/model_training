"""
Positional Log Parser, Location Dissector, and Masking Engine for Plan 0 Protocol.
Handles raw BGL logs with fine-grained positional parsing, BlueGene/L hardware hierarchy,
and leakage-free masking regexes.
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any


@dataclass
class LocationInfo:
    raw_location: str
    rack: Optional[str] = None          # e.g. 'R02', 'R27'
    midplane: Optional[str] = None      # e.g. 'R02-M1' (rack-midplane)
    node_card: Optional[str] = None     # e.g. 'R02-M1-N0' (rack-midplane-nodecard)
    card_type: Optional[str] = None     # 'C' (Compute), 'I' (I/O), or 'L' (Linkcard)
    chip: Optional[str] = None          # e.g. 'J12-U11'
    is_service_level: bool = False      # True for NULL, '-', or non-hardware system lines

    def midplane_id(self) -> str:
        """Returns midplane identifier or 'SERVICE' for incident clustering."""
        if self.is_service_level or not self.midplane:
            return "SERVICE"
        return self.midplane


@dataclass
class BGLRecord:
    line_idx: int
    raw_label: str             # '-' or alert tag e.g. 'APPREAD', 'KERNREC'
    is_anomaly: bool           # True if raw_label != '-'
    epoch: int                 # Unix timestamp
    date_str: str              # e.g. '2005.06.03'
    time_str: str              # e.g. '2005-06-03-15.42.50.363779'
    log_type: str              # e.g. 'RAS'
    component: str             # e.g. 'KERNEL', 'APP', 'MMCS'
    level: str                 # e.g. 'INFO', 'FATAL', 'WARNING'
    raw_message: str           # Raw unmasked log message
    masked_message: str        # Masked message (standard)
    masked_num_retained: str   # Masked message with numbers retained (ablation)
    location: LocationInfo     # Dissected location


# Pre-compiled regex patterns for performance
RE_STANDARD_NODE = re.compile(
    r'^R(?P<rack>[0-9A-Fa-f]{2})-M(?P<mid>[0-9A-Fa-f])-N(?P<nc>[0-9A-Fa-f])-?(?P<ctype>[CI]):(?P<chip>J\d+-U\d+)$'
)
RE_LINK_CARD = re.compile(
    r'^R(?P<rack>[0-9A-Fa-f]{2})-M(?P<mid>[0-9A-Fa-f])-L(?P<lc>[0-9A-Fa-f])-(?P<chip>U\d+)-(?P<ctype>[CI])$'
)
RE_NODE_CARD_ONLY = re.compile(
    r'^R(?P<rack>[0-9A-Fa-f]{2})-M(?P<mid>[0-9A-Fa-f])-N(?P<nc>[0-9A-Fa-f\w])(?:-(?P<ctype>[CI]))?$'
)
RE_MIDPLANE_ONLY = re.compile(
    r'^R(?P<rack>[0-9A-Fa-f]{2})-M(?P<mid>[0-9A-Fa-f])$'
)

# Masking regexes
RE_IN_MSG_NODE = re.compile(r'R\d{2}-M[0-9A-Fa-f]-[\w:\-]+')
RE_IP = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?\b')
RE_HEX = re.compile(r'0x[0-9a-fA-F]+')
RE_HEX_ADDR = re.compile(r'\b[0-9a-fA-F]{8,16}\b')
RE_CORE = re.compile(r'core\.\d+')
RE_PATH = re.compile(r'(?:/[a-zA-Z0-9_\-\.]+){2,}')
RE_LONG_NUM = re.compile(r'\b\d{4,}\b')


def parse_location(loc_str: str) -> LocationInfo:
    """
    Parses BGL location token into rack, midplane, node card, card type (C/I), and chip.
    NULL-location lines are tagged as service-level.
    """
    loc = loc_str.strip()
    if not loc or loc in ("NULL", "-", "UNKNOWN_LOCATION"):
        return LocationInfo(
            raw_location=loc,
            is_service_level=True,
        )

    # 1. Standard Compute/IO Node: e.g. R02-M1-N0-C:J12-U11
    m = RE_STANDARD_NODE.match(loc)
    if m:
        rack_id = f"R{m.group('rack')}"
        mid_id = f"{rack_id}-M{m.group('mid')}"
        nc_id = f"{mid_id}-N{m.group('nc')}"
        return LocationInfo(
            raw_location=loc,
            rack=rack_id,
            midplane=mid_id,
            node_card=nc_id,
            card_type=m.group('ctype'),
            chip=m.group('chip'),
            is_service_level=False,
        )

    # 2. Link Card / Switch: e.g. R27-M1-L3-U18-C
    m = RE_LINK_CARD.match(loc)
    if m:
        rack_id = f"R{m.group('rack')}"
        mid_id = f"{rack_id}-M{m.group('mid')}"
        nc_id = f"{mid_id}-L{m.group('lc')}"
        return LocationInfo(
            raw_location=loc,
            rack=rack_id,
            midplane=mid_id,
            node_card=nc_id,
            card_type='L',
            chip=f"{m.group('chip')}-{m.group('ctype')}",
            is_service_level=False,
        )

    # 3. Node card without chip: e.g. R33-M1-ND
    m = RE_NODE_CARD_ONLY.match(loc)
    if m:
        rack_id = f"R{m.group('rack')}"
        mid_id = f"{rack_id}-M{m.group('mid')}"
        nc_id = f"{mid_id}-N{m.group('nc')}"
        return LocationInfo(
            raw_location=loc,
            rack=rack_id,
            midplane=mid_id,
            node_card=nc_id,
            card_type=m.group('ctype'),
            chip=None,
            is_service_level=False,
        )

    # 4. Midplane only
    m = RE_MIDPLANE_ONLY.match(loc)
    if m:
        rack_id = f"R{m.group('rack')}"
        mid_id = f"{rack_id}-M{m.group('mid')}"
        return LocationInfo(
            raw_location=loc,
            rack=rack_id,
            midplane=mid_id,
            node_card=None,
            card_type=None,
            chip=None,
            is_service_level=False,
        )

    # 5. Fallback for any rack prefix
    if loc.startswith('R') and len(loc) >= 3:
        rack_id = loc[:3]
        mid_id = loc[:6] if len(loc) >= 6 and loc[3:5] == '-M' else None
        return LocationInfo(
            raw_location=loc,
            rack=rack_id,
            midplane=mid_id,
            node_card=None,
            card_type=None,
            chip=None,
            is_service_level=False,
        )

    # Anything else is service-level
    return LocationInfo(
        raw_location=loc,
        is_service_level=True,
    )


def mask_log_message(msg: str, keep_numbers: bool = False) -> str:
    """
    Masks dynamic parameters in a BGL log message:
    - In-message node IDs -> <NODE>
    - IP addresses -> <IP>
    - Hex numbers & addresses -> <HEX>
    - core.N -> <CORE>
    - File/directory paths -> <PATH>
    - Long numbers (>= 4 digits) -> <NUM> (unless keep_numbers=True for ablation)
    """
    # Order matters: replace specific identifiers before generic numbers
    msg = RE_IN_MSG_NODE.sub('<NODE>', msg)
    msg = RE_IP.sub('<IP>', msg)
    msg = RE_HEX.sub('<HEX>', msg)
    msg = RE_CORE.sub('<CORE>', msg)
    msg = RE_PATH.sub('<PATH>', msg)
    
    if not keep_numbers:
        msg = RE_HEX_ADDR.sub('<HEX>', msg)
        msg = RE_LONG_NUM.sub('<NUM>', msg)
    
    return msg


def parse_bgl_line(raw_line: str, line_idx: int = 0) -> Optional[BGLRecord]:
    """
    Parses a single BGL line strictly by positional columns:
    [0] Label: '-' or alert category (e.g. 'APPREAD', 'KERNREC')
    [1] Epoch: unix timestamp integer
    [2] Date: 'YYYY.MM.DD'
    [3] Node: hardware location string (or 'NULL')
    [4] Time: formatted timestamp 'YYYY-MM-DD-hh.mm.ss.uuuuuu'
    [5] Repeat node: redundant location string
    [6] Type: subsystem e.g. 'RAS'
    [7] Component: subsystem component e.g. 'KERNEL', 'APP', 'MMCS'
    [8] Level: log level e.g. 'INFO', 'FATAL', 'WARNING'
    [9+] Message: payload text
    """
    parts = raw_line.strip().split()
    if len(parts) < 9:
        return None

    raw_label = parts[0]
    is_anomaly = (raw_label != "-")
    try:
        epoch = int(parts[1])
    except ValueError:
        epoch = 0

    date_str = parts[2]
    raw_node = parts[3]
    time_str = parts[4]
    log_type = parts[6]
    component = parts[7]
    level = parts[8]
    raw_message = " ".join(parts[9:]) if len(parts) > 9 else ""

    location_info = parse_location(raw_node)
    masked_std = mask_log_message(raw_message, keep_numbers=False)
    masked_num = mask_log_message(raw_message, keep_numbers=True)

    return BGLRecord(
        line_idx=line_idx,
        raw_label=raw_label,
        is_anomaly=is_anomaly,
        epoch=epoch,
        date_str=date_str,
        time_str=time_str,
        log_type=log_type,
        component=component,
        level=level,
        raw_message=raw_message,
        masked_message=masked_std,
        masked_num_retained=masked_num,
        location=location_info,
    )
