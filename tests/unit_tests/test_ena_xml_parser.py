"""Unit tests for ENA XML Parser module using snapshot testing."""

from pathlib import Path

from langchain_omics.utils.ena_xml_parser import parse_ena_xml


# Get the path to the fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_xml_file(filename: str) -> str:
    """Load XML content from fixtures file."""
    file_path = FIXTURES_DIR / filename
    with open(file_path, "r") as f:
        return f.read()


def test_parse_study_xml(snapshot):
    """Test parsing STUDY XML data."""
    xml_content = load_xml_file("study_example.xml")
    result = parse_ena_xml(xml_content)
    assert result == snapshot


def test_parse_sample_xml(snapshot):
    """Test parsing SAMPLE XML data."""
    xml_content = load_xml_file("sample_example.xml")
    result = parse_ena_xml(xml_content)
    assert result == snapshot


def test_parse_experiment_xml(snapshot):
    """Test parsing EXPERIMENT XML data."""
    xml_content = load_xml_file("experiment_example.xml")
    result = parse_ena_xml(xml_content)
    assert result == snapshot


def test_parse_run_xml(snapshot):
    """Test parsing RUN XML data."""
    xml_content = load_xml_file("run_example.xml")
    result = parse_ena_xml(xml_content)
    assert result == snapshot