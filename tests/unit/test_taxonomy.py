"""Unit tests for Milestone 2 taxonomy extensions (job titles, locations)."""

from datetime import date
from ai_core.schemas import CVProfile, Experience
from ai_core.taxonomy import extract_job_titles, infer_location, normalize_location


def test_normalize_location():
    assert normalize_location("Ha Noi") == "Hà Nội"
    assert normalize_location("TP.HCM") == "Hồ Chí Minh"
    assert normalize_location("Nha Trang") == "Khánh Hòa"
    assert normalize_location("Bắc Ninh") == "Bắc Ninh"


def test_infer_location():
    prof1 = CVProfile(address="Quận Cầu Giấy, Ha Noi")
    assert infer_location(prof1) == "Hà Nội"

    prof2 = CVProfile(experiences=[Experience(job_title="Dev", location="District 1, HCM")])
    assert infer_location(prof2) == "Hồ Chí Minh"


def test_extract_job_titles():
    prof = CVProfile(
        headline="Senior Backend Engineer",
        experiences=[
            Experience(job_title="Python Developer"),
            Experience(job_title="Data Engineer"),
        ],
    )
    titles = extract_job_titles(prof)
    assert "Backend Developer" in titles
    assert "Software Engineer" in titles
    assert "Data Engineer" in titles
