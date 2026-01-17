"""ENA XML Parser for converting ENA Browser API XML responses to JSON-serializable dictionaries.

This module provides functionality to parse XML data from the ENA Browser API endpoint
(https://www.ebi.ac.uk/ena/browser/api/xml/) and convert it to a structured JSON format.

The parser handles four entity types: Study, Sample, Experiment, and Run, following
specific parsing rules:
- Root element <{TYPE}_SET> contains <{TYPE}> child elements
- Tag names are converted to lowercase
- Attributes are mapped to dictionary keys
- Text content is stored in a "value" field
- <*_LINKS> tags are skipped
- <*_ATTRIBUTES> tags are parsed to extract <TAG>/<VALUE> pairs
"""

import xml.etree.ElementTree as ET
from typing import Any, Dict, List


def parse_ena_xml(xml_string: str) -> Dict[str,Dict[str, Any]]:
    """Parse ENA XML string and convert to JSON-serializable list of dictionaries.

    This is the main entry point for parsing ENA XML data. It handles all four
    entity types (Study, Sample, Experiment, Run) by detecting the root element
    and applying the appropriate parsing logic.

    Args:
        xml_string: XML string from ENA Browser API

    Returns:
        List of dictionaries, one per entity (STUDY, SAMPLE, EXPERIMENT, or RUN)

    Raises:
        ValueError: If XML parsing fails or root element is invalid

    Example:
        >>> xml_data = '<STUDY_SET><STUDY accession="SRP001709">...</STUDY></STUDY_SET>'
        >>> result = parse_ena_xml(xml_data)
        >>> print(result[0]['accession'])
        'SRP001709'
    """
    root = parse_xml_string(xml_string)
    entity_type = find_root_element(root)
    
    # Get the child elements (e.g., <STUDY> elements from <STUDY_SET>)
    type_elements = list(root)
    
    result = {}
    for element in type_elements:
        parsed = parse_type_element(element, entity_type)
        result[parsed["accession"]] = parsed
    
    return result


def parse_xml_string(xml_string: str) -> ET.Element:
    """Parse XML string and return root element.

    Args:
        xml_string: XML string to parse

    Returns:
        Root XML element

    Raises:
        ValueError: If XML parsing fails
    """
    try:
        return ET.fromstring(xml_string)
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse XML: {e}")


def find_root_element(root: ET.Element) -> str:
    """Extract entity type from root element tag.

    Args:
        root: Root XML element

    Returns:
        Entity type (e.g., "STUDY", "SAMPLE", "EXPERIMENT", "RUN")

    Raises:
        ValueError: If root element tag doesn't match expected pattern
    """
    tag = root.tag
    if not tag.endswith("_SET"):
        raise ValueError(f"Invalid root element: {tag}. Expected <{{TYPE}}_SET>")
    
    entity_type = tag[:-4]  # Remove "_SET" suffix
    return entity_type


def parse_type_element(element: ET.Element, entity_type: str) -> Dict[str, Any]:
    """Parse a single TYPE element (e.g., <STUDY>, <SAMPLE>).

    Extracts root-level attributes and recursively parses child elements
    into a nested "value" dictionary.

    Args:
        element: XML element representing a single entity
        entity_type: Type of entity (e.g., "STUDY", "SAMPLE")

    Returns:
        Dictionary with root attributes and nested "value" containing children
    """
    result: Dict[str, Any] = {}
    
    # Extract root-level attributes (accession, alias, center_name, broker_name)
    for attr_name, attr_value in element.attrib.items():
        result[attr_name] = attr_value
    
    # Parse child elements recursively
    children_dict = parse_children_recursive(element)
    
    # Only add "value" if there are children
    if children_dict:
        result["value"] = children_dict
    
    return result


def parse_children_recursive(element: ET.Element) -> Dict[str, Any]:
    """Recursively parse child elements of an XML element.

    Handles special cases:
    - Skips <*_LINKS> tags
    - Parses <*_ATTRIBUTES> tags to extract TAG/VALUE pairs
    - Recursively parses other elements

    Args:
        element: XML element whose children to parse

    Returns:
        Dictionary mapping lowercase tag names to parsed values
    """
    result: Dict[str, Any] = {}
    
    for child in element:
        tag = child.tag
        
        # Skip <*_LINKS> tags
        if tag.endswith("_LINKS"):
            continue
        
        # Handle <*_ATTRIBUTES> tags specially
        if tag.endswith("_ATTRIBUTES"):
            attributes_dict = parse_attributes_element(child)
            if attributes_dict:
                key = tag.lower()
                result[key] = attributes_dict
            continue
        
        # Parse regular element
        parsed = parse_element(child)
        key = tag.lower()
        result[key] = parsed
    
    return result


def parse_element(element: ET.Element) -> Dict[str, Any]:
    """Parse a single XML element.

    Extracts attributes as dictionary keys and text content as "value".
    Recursively parses child elements.

    Args:
        element: XML element to parse

    Returns:
        Dictionary with attributes, text value, and nested children
    """
    result: Dict[str, Any] = {}
    
    # Extract attributes
    for attr_name, attr_value in element.attrib.items():
        result[attr_name] = attr_value
    
    # Extract text content
    text = element.text
    if text is not None:
        text = text.strip()
    
    # Recursively parse children
    children_dict = parse_children_recursive(element)
    
    # Simplification: if element has only text content and no attributes or children,
    # return just the text value
    if not element.attrib and not children_dict and text:
        return text
    
    # Simplification: if element has only one child that is self-closing
    # (no text, no attributes, no children), use child's tag name as value
    if not element.attrib and not text and len(children_dict) == 1:
        child_key, child_value = next(iter(children_dict.items()))
        if isinstance(child_value, dict) and not child_value:
            # Child is an empty dict (self-closing tag)
            return child_key
    
    # Otherwise, build up full dictionary
    if text:
        result["value"] = text
    
    # Merge children with result
    for key, value in children_dict.items():
        result[key] = value
    
    return result


def parse_attributes_element(element: ET.Element) -> Dict[str, str]:
    """Parse <*_ATTRIBUTES> element to extract TAG/VALUE pairs.

    Each child element (e.g., <STUDY_ATTRIBUTE>) contains <TAG> and <VALUE>
    elements that are extracted as key-value pairs.

    Args:
        element: XML element representing *_ATTRIBUTES

    Returns:
        Dictionary mapping TAG values to VALUE values

    Example:
        Input XML:
            <STUDY_ATTRIBUTES>
                <STUDY_ATTRIBUTE>
                    <TAG>parent_bioproject</TAG>
                    <VALUE>PRJNA46941</VALUE>
                </STUDY_ATTRIBUTE>
            </STUDY_ATTRIBUTES>

        Output:
            {"parent_bioproject": "PRJNA46941"}
    """
    result: Dict[str, str] = {}
    
    for attribute_element in element:
        tag = None
        value = None
        
        for child in attribute_element:
            if child.tag == "TAG":
                tag = child.text
            elif child.tag == "VALUE":
                value = child.text
        
        if tag is not None and value is not None:
            result[tag] = value
    
    return result