"""Unified SRA client using EBI Search and ENA Portal APIs."""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlencode

import requests
from pydantic import BaseModel, Field, validate_call
from typing_extensions import Annotated

from .ena_xml_parser import parse_ena_xml


@dataclass(frozen=True)
class Entry:
    """Represents an EBI entry with an accession and domain.
    
    Attributes:
        acc: The unique accession identifier of the entry.
        domain: The EBI domain type (e.g., 'sra-study', 'sra-experiment').
    """
    acc: str
    domain: str
    
    def __eq__(self, other: object) -> bool:
        """Compare entries based on their accession only.
        
        Args:
            other: Another object to compare with.
            
        Returns:
            True if both are Entry objects with the same accession.
        """
        if not isinstance(other, Entry):
            return NotImplemented
        return self.acc == other.acc
    
    def __hash__(self) -> int:
        """Hash entry based on its accession only.
        
        Returns:
            Hash value based on the accession.
        """
        return hash(self.acc)

class UndirectedGraph:
    """Undirected graph for storing Entry relationships."""
    
    def __init__(self) -> None:
        self._adj: Dict[Entry, Set[Entry]] = {}
    
    def add_edge(self, n1: Entry, n2: Entry) -> None:
        """Add an edge between two entries."""
        if n1 not in self._adj:
            self._adj[n1] = set()
        if n2 not in self._adj:
            self._adj[n2] = set()
        self._adj[n1].add(n2)
        self._adj[n2].add(n1)
    
    def has_edge(self, n1: Entry, n2: Entry) -> bool:
        """Check if an edge exists between two entries."""
        return n2 in self._adj.get(n1, set())
    
    def get_neighbors(self, entry: Entry) -> List[Entry]:
        """Get all neighboring entries for a given entry."""
        return list(self._adj.get(entry, []))
    
    def get_linked_entries(self) -> List[Entry]:
        """Get all entries that have at least one connection."""
        return list(self._adj.keys())

class UnifiedSRAClientConfig(BaseModel):
    """Private configuration class for UnifiedSRAClient with field validation."""
    
    request_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Timeout in seconds for HTTP requests"
    )
    rate_limit_delay: float = Field(
        default=0,
        ge=0,
        le=60,
        description="Delay in seconds between requests for rate limiting"
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        le=100,
        description="Maximum number of entries per batch request"
    )
    search_size: int = Field(
        default=100,
        ge=1,
        le=100,
        description="Number of results to fetch per page in search/crossref operations"
    )
    default_max_crossref: int = Field(
        default=1000,
        ge=1,
        le=1000000,
        description="Default maximum number of cross-references to return per entry"
    )
    default_max_search_hits: int = Field(
        default=100,
        ge=1,
        le=1000000,
        description="Default maximum number of search results to return"
    )
    log_level: str = Field(
        default="WARNING",
        pattern="^(DEBUG|INFO|WARNING|ERROR)$",
        description="Logging level - DEBUG, INFO, WARNING, or ERROR"
    )

class UnifiedSRAClient:
    """Unified SRA client using EBI Search and ENA Browser APIs.

    This client combines the strengths of both APIs:
    - EBI Search: Simple queries, efficient cross-references
    - ENA Browser: Full XML data retrieval with rich nested structures

    Attributes:
        _config: Private configuration object with validated parameters
    """

    _ebi_search_base_url: str = "https://www.ebi.ac.uk/ebisearch/ws/rest"
    _ena_portal_base_url: str = "https://www.ebi.ac.uk/ena/portal/api"
    _ena_browser_base_url: str = "https://www.ebi.ac.uk/ena/browser/api"

    _headers: Dict[str, str] = {
        "User-Agent": "langchain-omics"
    }

    _relationships: Dict[str, List[str]] = {
        "sra-study": ["sra-experiment"],
        "sra-experiment": ["sra-study", "sra-sample", "sra-run"],
        "sra-sample": ["sra-experiment"],
        "sra-run": ["sra-experiment"]
    }

    _valid_domains: Set[str] = {"sra-study", "sra-experiment", "sra-sample", "sra-run"}

    def __init__(
        self,
        config: Optional[UnifiedSRAClientConfig] = None
    ):
        """Initialize UnifiedSRAClient with validated configuration."""
        self._config = config or UnifiedSRAClientConfig()
        
        # Setup logging
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.setLevel(getattr(logging, self._config.log_level))
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)


    def _make_request(
            self, url: str, params: Optional[Dict[str, Any]] = None,
            mimetype: str = "json"
        ) -> Any:
        """Make HTTP request with error handling and rate limiting.
        
        Args:
            url: Request URL
            params: Optional query parameters
            mimetype: Response type - 'xml' or 'json' (default: 'json')
        
        Returns:
            JSON response or raw XML text if mimetype='xml'
        """
        if params:
            clean_params = {k: str(v) for k, v in params.items() if v is not None}
            if clean_params:
                url = f"{url}?{urlencode(clean_params)}"

        self._logger.debug(f"Making request to {url} with mimetype={mimetype}")

        time.sleep(self._config.rate_limit_delay)

        # Set appropriate headers based on mimetype
        headers = dict(self._headers)
        if mimetype == "xml":
            headers["Accept"] = "application/xml"
        elif mimetype == "json":
            headers["Accept"] = "application/json"
        else:
            raise ValueError(f"Invalid mimetype: {mimetype}. Must be 'xml' or 'json'")

        response = requests.get(
            url, headers=headers, timeout=self._config.request_timeout
        )

        # Handle 204 No Content (end of pagination)
        if response.status_code == 204:
            return [] if mimetype == "json" else ""

        response.raise_for_status()
        
        if mimetype == "xml":
            return response.text
        else:
            return response.json()

    def _fetch_entries_batch(
        self, accs: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Use ENA Browser XML API for entry retrieval (private batch method).

        Args:
            accs: List of accession identifiers (max 100 per batch)

        Returns:
            Dict of entry dictionaries with full XML-parsed data, keyed by accession
        """
        if not accs:
            return dict()

        # Limit to 100 entries per batch (ENA Browser limit)
        if len(accs) > self._config.batch_size:
            raise ValueError(
                f"Batch size {len(accs)} exceeds maximum allowed batch size "
                f"of {self._config.batch_size}"
            )

        self._logger.debug(
            f"Getting entries batch: {len(accs)} entries"
        )

        # Build URL for ENA Browser XML API
        url = f"{self._ena_browser_base_url}/xml/{','.join(accs)}"
        params = {
            "download": "false",
            "gzip": "true",
            "includeLinks": "false"
        }

        # Fetch XML data from ENA Browser API
        xml_response = self._make_request(url, params, mimetype="xml")

        # Parse XML using the XML parser
        parsed_dict = parse_ena_xml(xml_response)
        self._logger.debug(f"Parsed XML into {len(parsed_dict)} entries")
        return parsed_dict

    @validate_call
    def fetch_entries(
        self,
        accs: Annotated[
            List[str],
            Field(min_length=1, description="List of accession identifiers to fetch")
        ]
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch entries in batches of 100 (ENA Browser limit).

        Args:
            accs: List of accession identifiers (any size)

        Returns:
            Dict of all entry dictionaries, keyed by accession
        """
        if not accs:
            return dict()

        self._logger.debug(
            f"Getting entries: {len(accs)} total entries"
        )
        all_entries = dict()

        # Process in batches of self.batch_size
        for i in range(0, len(accs), self._config.batch_size):
            batch_accs = accs[i : i + self._config.batch_size]
            self._logger.debug(
                f"Processing batch from {i}: {len(batch_accs)} entries"
            )
            batch_entries = self._fetch_entries_batch(batch_accs)
            all_entries.update(batch_entries)

        self._logger.debug(f"Retrieved {len(all_entries)} total entries")
        return all_entries

    def _fetch_crossref_batch(
        self,
        domain: str,
        accs: List[str],
        target_domain: str,
        max_size: Optional[int] = None
    ) -> Dict[str, Set[str]]:
        """Use EBI Search for cross-references with pagination support.

        Args:
            domain: Source domain (e.g., 'sra-sample')
            accs: List of source accession identifiers (max self._config.batch_size)
            target_domain: Target domain (e.g., 'sra-experiment')
            max_size: Maximum number of cross-references to return per entry 
                (default: default_max_crossref)

        Returns:
            Dictionary mapping accession -> set of cross-referenced accessions
        """
        if not accs:
            return {}

        # Limit to 100 entries per batch
        if len(accs) > self._config.batch_size:
            raise ValueError(
                f"Batch size {len(accs)} exceeds maximum allowed batch size "
                f"of {self._config.batch_size}"
            )

        if max_size is None:
            max_size = self._config.default_max_crossref

        all_results = {}
        accs_to_process = list(accs)
        start: int = 0

        # Process entries with pagination support
        # The loop handles paginated cross-reference results, fetching additional
        # pages when the API reports more references than returned in a single response
        while len(accs_to_process) > 0 and start < max_size:

            url = (
                f"{self._ebi_search_base_url}/{domain}/entry/"
                f"{','.join(accs_to_process)}/xref/{target_domain}"
            )
            
            # Clear the list - we'll repopulate if more pages are needed
            accs_to_process = []

            if start + self._config.search_size > max_size:
                search_size = max_size - start
            else:
                search_size = self._config.search_size

            # Use search_size for the API request (page size)
            params = {"format": "json", "size": str(search_size), "start": str(start)}

            self._logger.debug(
                f"Cross-reference batch call: {domain} -> {target_domain}, "
                f"{len(accs)} entries, max_size={max_size}"
            )
        
            response = self._make_request(url, params)
            response_entries = response.get("entries", [])

            # Build mapping of source_id -> [target_ids] and check for pagination
            for response_entry in response_entries:
                source_acc = response_entry.get("acc") or response_entry.get("id")
                if source_acc:
                    cross_refs = set()
                    references = response_entry.get("references", [])
                    for ref in references:
                        acc = ref.get("acc")
                        if acc:
                            cross_refs.add(acc)
                    if source_acc not in all_results:
                        all_results[source_acc] = cross_refs
                    else:
                        all_results[source_acc] |= cross_refs

                    reference_count = response_entry.get("referenceCount", 0)
                    # Check if there are more references to fetch (pagination)
                    if start + len(cross_refs) < reference_count:
                        accs_to_process.append(source_acc)


            # Increment start AFTER processing the request
            start = start + self._config.search_size

        # Add empty sets for entries that were not returned
        for acc in accs:
            if acc not in all_results:
                all_results[acc] = set()

        return all_results

    @validate_call
    def fetch_crossref(
        self,
        domain: Annotated[
            str,
            Field(
                pattern=r"^sra-(study|experiment|sample|run)$",
                description="SRA domain name"
            )
        ],
        accs: Annotated[
            List[str],
            Field(
                min_length=1,
                description="List of source accession identifiers"
            )
        ],
        target_domain: Annotated[
            str,
            Field(
                pattern=r"^sra-(study|experiment|sample|run)$",
                description="Target SRA domain name"
            )
        ],
        max_size: Annotated[
            Optional[int],
            Field(
                ge=1,
                le=1000000,
                description="Maximum cross-references per entry"
            )
        ] = None
    ) -> Dict[str, Set[str]]:
        """Use EBI Search for cross-references with pagination support.

        Args:
            domain: Source domain (e.g., 'sra-sample')
            accs: List of source accession identifiers (any size)
            target_domain: Target domain (e.g., 'sra-experiment')
            max_size: Maximum number of cross-references to return per entry (default:
                default_max_crossref)

        Returns:
            Dictionary mapping accession -> set of cross-referenced accessions
        """
        if not accs:
            return {}

        all_results = {}

        # Process in chunks of self._config.batch_size
        for i in range(0, len(accs), self._config.batch_size):
            batch_accs = accs[i : i + self._config.batch_size]
            batch_results = self._fetch_crossref_batch(
                domain, batch_accs, target_domain, max_size
            )
            all_results.update(batch_results)

        return all_results

    @validate_call
    def search_domain(
        self,
        domain: Annotated[
            str,
            Field(
                pattern=r"^sra-(study|experiment|sample|run)$",
                description="SRA domain name"
            )
        ],
        query: Annotated[
            str,
            Field(
                min_length=1,
                description="Search query string"
            )
        ],
        max_size: Annotated[
            Optional[int],
            Field(
                ge=1,
                le=1000000,
                description="Maximum number of results to return"
            )
        ] = None
    ) -> List[str]:
        """Use EBI Search for simple domain search with pagination support.

        Args:
            domain: SRA domain (e.g., 'sra-study', 'sra-sample')
            query: Search query string
            max_size: Maximum number of results to return (default:
                default_max_search_hits)

        Returns:
            List of accession identifiers
        """
        if max_size is None:
            max_size = self._config.default_max_search_hits

        self._logger.debug(
            f"Searching domain {domain} with query '{query}', max_size={max_size}"
        )

        returned_accs = []
        start = 0

        while len(returned_accs) < max_size:
            # Determine how many results to fetch in this request
            fetch_size = min(self._config.search_size, max_size - len(returned_accs))

            url = f"{self._ebi_search_base_url}/{domain}"
            params = {
                "query": query,
                "size": str(fetch_size),
                "start": str(start),
                "format": "json"
            }

            response = self._make_request(url, params)
            response_entries = response.get("entries", [])
            hit_count = response.get("hitCount", 0)

            if not response_entries:
                # No more results available
                break

            # Extract accessions from response entries
            batch_accs : List[str] = [
                response_entry.get("acc") or response_entry.get("id")
                for response_entry in response_entries
                if response_entry.get("acc") or response_entry.get("id")
            ]
            returned_accs.extend(batch_accs)
            self._logger.debug(
                f"Fetched {len(batch_accs)} entries"
                f"(total: {len(returned_accs)}/{max_size}, "
                f"hitCount: {hit_count})"
            )

            # Check if we've fetched all available results
            if start + len(response_entries) >= hit_count:
                break

            start += len(response_entries)

        self._logger.debug(
            f"Domain search returned {len(returned_accs)} entries for {domain}"
        )
        return returned_accs

    @validate_call
    def search_multiple_domains(
        self,
        domains: Annotated[
            List[str],
            Field(
                min_length=1,
                description="List of EBI domain names to search"
            )
        ],
        query: Annotated[
            str,
            Field(
                min_length=1,
                description="Search query string"
            )
        ],
        max_size: Annotated[
            Optional[int],
            Field(
                ge=1,
                le=1000000,
                description="Default maximum results per domain"
            )
        ] = None,
        max_size_per_domain: Annotated[
            Optional[Dict[str, int]],
            Field(description="Custom max_size per domain")
        ] = None,
    ) -> Dict[str, List[str]]:
        """Search across multiple EBI domains and return results by domain.

        Args:
            domains: List of EBI domain names to search
                (e.g., ["sra-study", "sra-sample"])
            query: Search query string
            max_size: Default maximum number of results per domain
                (used if not in max_size_per_domain, default: default_max_search_hits)
            max_size_per_domain: Optional dictionary mapping domain name to custom 
                max_size limit
                If domain is not in keys, uses the default max_size parameter

        Returns:
            Dictionary mapping domain name to list of accessions.
            Maintains search order by processing domains in the order provided.
        """
        self._logger.debug(f"Searching multiple domains: {domains}")
        results = {}

        for domain in domains:
            if domain not in self._valid_domains:
                raise ValueError(
                    f"Invalid domain: '{domain}'. "
                    f"Must be one of: {', '.join(self._valid_domains)}"
                )
            
            # Determine effective max_size for this domain
            effective_max_size = max_size
            if effective_max_size is None:
                effective_max_size = self._config.default_max_search_hits
            
            if max_size_per_domain:
                domain_max_size = max_size_per_domain.get(domain, effective_max_size)
            else:
                domain_max_size = effective_max_size

            self._logger.debug(
                f"Searching domain: {domain} with max_size={domain_max_size}"
            )
            returned_accs = self.search_domain(domain, query, max_size=domain_max_size)
            results[domain] = returned_accs
            self._logger.debug(f"Found {len(returned_accs)} entries in {domain}")

        return results

    @validate_call
    def fetch_crossref_recursively(
        self,
        entries: Annotated[
            List[Entry],
            Field(
                min_length=1,
                description="List of Entry objects to start from"
            )
        ],
        max_depth: Annotated[
            int,
            Field(
                ge=1,
                le=10,
                description="Maximum recursion depth"
            )
        ] = 2,
        max_size: Annotated[
            Optional[int],
            Field(
                ge=1,
                le=1000000,
                description="Maximum cross-references per entry"
            )
        ] = None
    ) -> UndirectedGraph:
        """Recursively build cross-reference relations with depth limit.

        Args:
            entries: List of Entry objects to start from
            max_depth: Maximum recursion depth (default: 2)
            max_size: Maximum number of cross-references to fetch per entry 
            (default: default_max_crossref)

        Returns:
            UndirectedGraph: A graph object containing all cross-reference relationships
                discovered during the recursive traversal.
        """

        graph = UndirectedGraph()

        def _recursive_walk(
            _graph: UndirectedGraph,
            _entries_by_domain: Dict[str, Set[str]],
            _fetched: Dict[str, Set[str]],
            _depth: int,
        ) -> UndirectedGraph:
            """Recursive helper function to traverse cross-references and build 
                the graph.
            
            This implements a breadth-first traversal across domains, collecting
            cross-references up to the specified maximum depth. For each domain,
            it follows only the defined relationship paths (non-branching per domain).
            """
            # Base case: stop recursion if max depth was reached
            if _depth >= max_depth:
                return _graph
            
            # Initialize next_entries_by_domain with empty sets for all valid domains
            next_entries_by_domain = {}
            next_count: int = 0
            for domain in self._valid_domains:
                next_entries_by_domain[domain] = set()

            # Iterate over all source domains to find cross-references
            for domain, queries in _entries_by_domain.items():

                # Skip if domain has no defined relationships
                if domain not in self._relationships:
                    raise ValueError("Invalid domain: " + domain)

                # For each source domain, follow its defined relationship paths
                for target_domain in self._relationships[domain]:

                    crossref = self.fetch_crossref(
                        domain, list(queries), target_domain, max_size)
                    
                    # Track already-fetched entries to avoid duplicates
                    _fetched[domain] |= queries
                    
                    for source_acc, target_accs in crossref.items():
                        # Collect new entries for next recursion level
                        next_count += len(target_accs)
                        new_entries = target_accs - _fetched[target_domain]
                        next_entries_by_domain[target_domain] = \
                            next_entries_by_domain[target_domain] | new_entries
                        
                        # Add edges to graph for all cross-references
                        for target_acc in target_accs:
                            source_entry = Entry(acc=source_acc, domain=domain)
                            target_entry = Entry(acc=target_acc, domain=target_domain)
                            _graph.add_edge(source_entry, target_entry)
            # Continue to next recursion only if there are new entries to fetch
            if next_count > 0:
                _graph = _recursive_walk(
                    _graph, next_entries_by_domain, _fetched, _depth + 1
                )
            return _graph
        
        # Initialize fetched dictionary with empty sets for each domain
        init_fetched = {}
        for domain in self._valid_domains:
            init_fetched[domain] = set()
        
        # Initialize entries_by_domain with empty sets, then populate with input entries
        init_entries_by_domain = {}
        for domain in self._valid_domains:
            init_entries_by_domain[domain] = set()
        for entry in entries:
            init_entries_by_domain[entry.domain].add(entry.acc)
        
        graph = _recursive_walk(graph, init_entries_by_domain, init_fetched, 0)
        return graph
