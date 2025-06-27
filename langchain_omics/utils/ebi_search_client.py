"""EBI Search API client for langchain-omics."""

import time
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlencode

import requests
from langchain_core.documents import Document
from pydantic import BaseModel, Field, ConfigDict

class EBISearchAPIWrapper(BaseModel):
    """Wrapper around EBI Search API.
    
    This class provides methods to interact with the EBI Search REST API,
    including searching domains, retrieving entry details, and getting
    cross-references between different databases.
    """
    
    base_url: str = "https://www.ebi.ac.uk/ebisearch/ws/rest/"
    max_k: int = 10
    page_size: int = 25
    headers: Dict[str, str] = Field(
        default_factory=lambda: {
            "User-Agent": "langchain-omics/1.0.0",
            "Accept": "application/json",
        }
    )
    request_timeout: int = 30
    rate_limit_delay: float = 0.1  # Delay between requests in seconds
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _make_request(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make HTTP request to EBI Search API with error handling.
        
        Args:
            url: The API endpoint URL
            params: Query parameters
            
        Returns:
            JSON response as dictionary
            
        Raises:
            requests.RequestException: If the request fails
            ValueError: If the response is not valid JSON
        """
        if params:
            # Clean up None values and convert to strings
            clean_params = {k: str(v) for k, v in params.items() if v is not None}
            if clean_params:
                url = f"{url}?{urlencode(clean_params)}"
        
        # Rate limiting
        time.sleep(self.rate_limit_delay)
        
        try:
            response = requests.get(
                url, 
                headers=self.headers, 
                timeout=self.request_timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise requests.RequestException(f"EBI Search API request failed: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid JSON response from EBI Search API: {e}")

    def _load(self, query: str, domain: str, size: Optional[int] = None, 
              fields: Optional[List[str]] = None, start: int = 0) -> List[Dict[str, Any]]:
        """Search EBI Search for documents matching the query in specified domain.
        
        Args:
            query: Search query string
            domain: Domain identifier (e.g., 'sra-sample', 'sra-study')
            size: Number of entries to retrieve (default: page_size)
            fields: List of field identifiers to retrieve
            start: Index of first entry in results
            
        Returns:
            List of entry dictionaries
        """
        if size is None:
            size = min(self.page_size, self.max_k)
        
        url = urljoin(self.base_url, domain)
        params = {
            "query": query,
            "size": str(size),
            "start": str(start),
            "format": "json"
        }
        
        if fields:
            params["fields"] = ",".join(fields)
        
        response_data = self._make_request(url, params)
        return response_data.get("entries", [])

    def _load_paginated(self, query: str, domain: str, total_needed: int, 
                       fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Load results page by page with fixed page size of 25.
        
        Args:
            query: Search query string
            domain: Domain identifier
            total_needed: Total number of entries needed
            fields: List of field identifiers to retrieve
            
        Returns:
            List of entry dictionaries up to total_needed
        """
        all_entries = []
        start = 0
        page_size = 25  # Fixed page size for efficient API usage
        
        while len(all_entries) < total_needed:
            remaining = total_needed - len(all_entries)
            current_page_size = min(page_size, remaining)
            
            page_entries = self._load(
                query=query,
                domain=domain,
                size=current_page_size,
                fields=fields,
                start=start
            )
            
            if not page_entries:
                break  # No more results available
                
            all_entries.extend(page_entries)
            start += len(page_entries)
            
            # If we got fewer than requested, we've hit the end
            if len(page_entries) < current_page_size:
                break
        
        return all_entries[:total_needed]

    def _get_entry_details(self, domain: str, entry_id: str, 
                          fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get detailed entry information with specified fields.
        
        Args:
            domain: Domain identifier
            entry_id: Entry identifier/accession
            fields: List of field identifiers to retrieve
            
        Returns:
            Entry dictionary with detailed information
        """
        url = urljoin(self.base_url, f"{domain}/entry/{entry_id}")
        params = {"format": "json"}
        
        if fields:
            params["fields"] = ",".join(fields)
        
        response_data = self._make_request(url, params)
        entries = response_data.get("entries", [])
        return entries[0] if entries else {}

    def _get_cross_references(self, domain: str, entry_id: str, target_domain: str,
                            size: int = 100, fields: Optional[List[str]] = None) -> List[Dict]:
        """Get cross-references from entry to target domain.
        
        Args:
            domain: Source domain identifier
            entry_id: Source entry identifier
            target_domain: Target domain to search for cross-references
            size: Maximum number of cross-references to retrieve
            fields: List of field identifiers to retrieve from target domain
            
        Returns:
            List of cross-referenced entry dictionaries
        """
        url = urljoin(self.base_url, f"{domain}/entry/{entry_id}/xref/{target_domain}")
        params = {
            "format": "json",
            "size": str(size)
        }
        
        if fields:
            params["fields"] = ",".join(fields)
        
        try:
            response_data = self._make_request(url, params)
            return response_data.get("entries", [])
        except Exception:
            # Cross-references may not exist, return empty list
            return []

    def _get_domain_metadata(self, domain: str) -> Dict[str, Any]:
        """Get metadata about a domain including available fields.
        
        Args:
            domain: Domain identifier
            
        Returns:
            Domain metadata dictionary
        """
        url = urljoin(self.base_url, domain)
        params = {"format": "json"}
        
        response_data = self._make_request(url, params)
        domains = response_data.get("domains", [])
        return domains[0] if domains else {}

    def _extract_field_value(self, entry: Dict, field_id: str) -> Optional[str]:
        """Extract field value from entry fields dictionary.
        
        Args:
            entry: Entry dictionary from EBI Search response
            field_id: Field identifier to extract
            
        Returns:
            Field value as string, or None if not found
        """
        fields = entry.get("fields", {})
        if isinstance(fields, dict):
            # New format: fields is a dictionary
            field_values = fields.get(field_id, [])
            return field_values[0] if field_values else None
        else:
            # Legacy format: fields is an array of objects
            for field in fields:
                if field.get("id") == field_id:
                    return field.get("firstvalue") or (field.get("values", [None])[0])
        return None

    def _extract_field_values(self, entry: Dict, field_id: str) -> List[str]:
        """Extract all field values from entry fields dictionary.
        
        Args:
            entry: Entry dictionary from EBI Search response
            field_id: Field identifier to extract
            
        Returns:
            List of field values
        """
        fields = entry.get("fields", {})
        if isinstance(fields, dict):
            # New format: fields is a dictionary
            return fields.get(field_id, [])
        else:
            # Legacy format: fields is an array of objects
            for field in fields:
                if field.get("id") == field_id:
                    return field.get("values", [])
        return []

    def _dict2document(self, entry: Dict[str, Any], domain: str, 
                      append_cross_ref: bool = True, fetch_details: bool = False,
                      domain_config: Optional[Dict] = None) -> Document:
        """Convert an entry dictionary to a Document object.
        
        Args:
            entry: Entry dictionary from EBI Search
            domain: Domain identifier
            append_cross_ref: Whether to append cross-ref info to text
            fetch_details: Whether to fetch additional details via API calls
            domain_config: Domain configuration dictionary
            
        Returns:
            Document object with formatted content and metadata
        """
        import json
        
        # Extract basic information
        acc = self._extract_field_value(entry, "acc") or entry.get("id", "")
        domain_name = domain_config.get("name", domain) if domain_config else domain
        
        # Prepare metadata
        metadata = {
            "primary_id": acc,
            "database": domain,
            "entity_type": domain_name,
            "source": "EBI Search",
            "url": f"https://www.ebi.ac.uk/ena/browser/view/{acc}",
            "fields": {},
            "cross_references": {},
            "related_entities": {}
        }
        
        # Extract all fields for metadata
        fields = entry.get("fields", {})
        if isinstance(fields, dict):
            # New format: fields is a dictionary
            for field_id, field_values in fields.items():
                if field_values:
                    metadata["fields"][field_id] = field_values[0] if len(field_values) == 1 else field_values
        else:
            # Legacy format: fields is an array of objects
            for field in fields:
                field_id = field.get("id")
                field_values = field.get("values", [])
                if field_id and field_values:
                    metadata["fields"][field_id] = field_values[0] if len(field_values) == 1 else field_values
        
        # ALWAYS extract cross-reference IDs (fast - from main response)
        if domain_config:
            cross_ref_ids = self._extract_cross_ref_ids(entry, domain_config)
            metadata["cross_references"] = cross_ref_ids
        
        # ONLY if fetch_details=True, make additional API calls
        if fetch_details and domain_config:
            cross_ref_details = self._resolve_cross_references(entry, domain_config)
            related_info = self._get_related_entities(entry, domain, domain_config.get("related_domains", []))
            metadata["cross_reference_details"] = cross_ref_details
            metadata["related_entities"] = related_info
        
        # Create structured page content from metadata
        content_data = {
            "id": acc,
            "type": domain_name,
            "database": domain,
            "fields": metadata["fields"],
            "cross_references": metadata["cross_references"]
        }
        
        # Add additional details if fetched
        if fetch_details:
            if metadata.get("cross_reference_details"):
                content_data["cross_reference_details"] = metadata["cross_reference_details"]
            if metadata.get("related_entities"):
                content_data["related_entities"] = metadata["related_entities"]
        
        # Format as JSON for page content
        page_content = json.dumps(content_data, indent=2, ensure_ascii=False)
        
        return Document(page_content=page_content, metadata=metadata)

    def _extract_cross_ref_ids(self, entry: Dict, domain_config: Dict) -> Dict:
        """Extract cross-reference IDs from main response (no API calls).
        
        Args:
            entry: Entry dictionary from EBI Search
            domain_config: Domain configuration dictionary
            
        Returns:
            Dictionary of cross-reference IDs
        """
        cross_refs = {}
        
        cross_ref_targets = domain_config.get("cross_ref_targets", {})
        for xref_field, target_domain in cross_ref_targets.items():
            xref_values = self._extract_field_values(entry, xref_field)
            if xref_values:
                cross_refs[xref_field] = xref_values
        
        return cross_refs

    def _resolve_cross_references(self, entry: Dict, domain_config: Dict) -> Dict:
        """Resolve cross-references based on domain configuration.
        
        Args:
            entry: Entry dictionary from EBI Search
            domain_config: Domain configuration dictionary
            
        Returns:
            Dictionary of cross-reference information
        """
        cross_refs = {}
        
        cross_ref_targets = domain_config.get("cross_ref_targets", {})
        for xref_field, target_domain in cross_ref_targets.items():
            xref_values = self._extract_field_values(entry, xref_field)
            if xref_values:
                cross_refs[xref_field] = xref_values
                
                # Try to get additional details for first cross-reference
                if xref_values and len(xref_values) > 0:
                    try:
                        xref_details = self._get_entry_details(
                            target_domain, 
                            xref_values[0], 
                            ["acc", "description", "organism"]
                        )
                        if xref_details:
                            cross_refs[f"{xref_field}_details"] = xref_details
                    except Exception:
                        # Cross-reference details may not be available
                        pass
        
        return cross_refs

    def _get_related_entities(self, entry: Dict, domain: str, related_domains: List[str]) -> Dict:
        """Get related entities from other domains.
        
        Args:
            entry: Entry dictionary from EBI Search
            domain: Source domain
            related_domains: List of related domain identifiers
            
        Returns:
            Dictionary of related entities by domain
        """
        related_entities = {}
        entry_id = self._extract_field_value(entry, "acc") or entry.get("id", "")
        
        if not entry_id:
            return related_entities
        
        for related_domain in related_domains:
            try:
                related_entries = self._get_cross_references(
                    domain, 
                    entry_id, 
                    related_domain,
                    size=5,  # Limit to 5 related entries
                    fields=["acc", "title", "description"]
                )
                if related_entries:
                    related_entities[related_domain] = related_entries
            except Exception:
                # Related entities may not exist
                continue
        
        return related_entities

    def get_max_k(self, query: str, domain: str) -> int:
        """Get the maximum number of results available for a query in a domain.
        
        Args:
            query: Search query string
            domain: Domain identifier
            
        Returns:
            Maximum number of results available for the query
        """
        try:
            # Make a minimal request to get hit count
            url = urljoin(self.base_url, domain)
            params = {
                "query": query,
                "size": "1",
                "format": "json"
            }
            
            response_data = self._make_request(url, params)
            return response_data.get("hitCount", 0)
            
        except Exception:
            # If we can't get the hit count, return 0
            return 0
