"""Location normalization taxonomy for 63 Vietnamese provinces and Hanoi wards."""

from __future__ import annotations

import re
from ai_core.schemas import CVProfile

TINH_THANH_VN: dict[str, list[str]] = {
    "Hà Nội": ["Hà Nội", "Ha Noi", "HaNoi", "HN"],
    "Hồ Chí Minh": ["Hồ Chí Minh", "Ho Chi Minh", "HoChiMinh", "TP.HCM", "TPHCM", "Sài Gòn", "Sai Gon", "HCM"],
    "Đà Nẵng": ["Đà Nẵng", "Da Nang", "DaNang"],
    "Cần Thơ": ["Cần Thơ", "Can Tho"],
    "Hải Phòng": ["Hải Phòng", "Hai Phong", "HaiPhong"],
    "An Giang": ["An Giang", "AnGiang"],
    "Bà Rịa - Vũng Tàu": ["Bà Rịa - Vũng Tàu", "Ba Ria - Vung Tau", "BaRiaVungTau", "Vũng Tàu"],
    "Bắc Giang": ["Bắc Giang", "Bac Giang", "BacGiang"],
    "Bắc Ninh": ["Bắc Ninh", "Bac Ninh", "BacNinh"],
    "Bình Dương": ["Bình Dương", "Binh Duong", "BinhDuong"],
    "Bình Định": ["Bình Định", "Binh Dinh", "BinhDinh"],
    "Bình Phước": ["Bình Phước", "Binh Phuoc"],
    "Bình Thuận": ["Bình Thuận", "Binh Thuan"],
    "Đồng Nai": ["Đồng Nai", "Dong Nai", "DongNai"],
    "Gia Lai": ["Gia Lai", "GiaLai"],
    "Hải Dương": ["Hải Dương", "Hai Duong"],
    "Khánh Hòa": ["Khánh Hòa", "Khanh Hoa", "KhanhHoa", "Nha Trang"],
    "Lâm Đồng": ["Lâm Đồng", "Lam Dong", "Đà Lạt", "Da Lat"],
    "Long An": ["Long An", "LongAn"],
    "Nghệ An": ["Nghệ An", "Nghe An", "Vinh"],
    "Quảng Nam": ["Quảng Nam", "Quang Nam"],
    "Quảng Ninh": ["Quảng Ninh", "Quang Ninh", "Hạ Long"],
    "Thái Nguyên": ["Thái Nguyên", "Thai Nguyen"],
    "Thanh Hóa": ["Thanh Hóa", "Thanh Hoa"],
    "Thừa Thiên Huế": ["Thừa Thiên Huế", "Thua Thien Hue", "Huế", "Hue"],
    "Vĩnh Phúc": ["Vĩnh Phúc", "Vinh Phuc"],
}

_LOCATION_LOOKUP: list[tuple[str, str]] = []
for canonical, variants in TINH_THANH_VN.items():
    for v in variants:
        _LOCATION_LOOKUP.append((v.lower(), canonical))
# Sort lookup by length descending so longer variants match first (e.g. "Ho Chi Minh" before "HCM")
_LOCATION_LOOKUP.sort(key=lambda x: -len(x[0]))


def normalize_location(text: str | None) -> str | None:
    """Normalize raw address/location string to canonical Vietnamese province name."""
    if not text:
        return None

    clean = text.strip()
    lower_text = clean.lower()

    for variant, canonical in _LOCATION_LOOKUP:
        pattern = rf"\b{re.escape(variant)}\b"
        if re.search(pattern, lower_text, re.IGNORECASE):
            return canonical

    return clean


def infer_location(profile: CVProfile) -> str | None:
    """Infer primary candidate location via priority chain:
    1. Canonicalized profile.address
    2. Canonicalized profile.experiences[-1].location (most recent company location)
    3. None
    """
    if profile.address:
        norm = normalize_location(profile.address)
        if norm:
            return norm

    for exp in profile.experiences:
        if exp.location:
            norm = normalize_location(exp.location)
            if norm:
                return norm

    return None
