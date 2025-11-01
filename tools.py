"""Tools that agents can use to perform their tasks.

Enhanced tools with:
- Robust error handling and retries
- HTML content extraction and cleaning
- Structured logging
- Fallback mechanisms
- Configurable timeouts and retry strategies
"""

import os
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    from crewai_tools import SerperDevTool
except ImportError:
    SerperDevTool = None

try:
    from crewai_tools import tool
except ImportError:
    # Fallback decorator if crewai_tools not available
    def tool(name: str):
        def decorator(func):
            func.tool_name = name
            return func
        return decorator


load_dotenv()

# Configuration (can be overridden via .env)
SCRAPE_TIMEOUT = int(os.getenv("SCRAPE_TIMEOUT", "25"))
SCRAPE_MAX_RETRIES = int(os.getenv("SCRAPE_MAX_RETRIES", "3"))
SCRAPER_USER_AGENT = os.getenv(
    "SCRAPER_USER_AGENT", 
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
ENABLE_TOOL_LOGGING = os.getenv("ENABLE_TOOL_LOGGING", "true").lower() == "true"
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", "50000"))  # chars

# Setup logging
log_dir = Path("runs") / "tool_logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"tools_{datetime.now().strftime('%Y%m%d')}.log"

logger = logging.getLogger("tools")
logger.setLevel(logging.INFO if ENABLE_TOOL_LOGGING else logging.WARNING)

if not logger.handlers:
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)


# Search tool - requires SERPER_API_KEY in .env
if SerperDevTool is not None:
    try:
        search_tool = SerperDevTool()
        logger.info("SerperDevTool initialized successfully")
    except Exception as e:
        search_tool = None
        logger.warning(f"Failed to initialize SerperDevTool: {e}")
else:
    search_tool = None
    logger.info("SerperDevTool not available (crewai_tools not installed or missing)")


def _create_session_with_retries(
    retries: int = 3,
    backoff_factor: float = 1.0,
    timeout: int = 25
) -> requests.Session:
    """Create a requests session with retry logic and connection pooling."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504, 408),
        allowed_methods=("GET", "HEAD", "OPTIONS"),
        raise_on_status=False  # We'll handle status codes manually
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=20
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _extract_text_from_html(html: str, max_length: Optional[int] = None) -> str:
    """Extract clean text from HTML, removing scripts, styles, and extra whitespace."""
    if not HAS_BS4:
        # Fallback: basic regex cleaning
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    else:
        # Use BeautifulSoup for better extraction
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
            element.decompose()
        
        # Extract text
        text = soup.get_text(separator=' ', strip=True)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    
    if max_length and len(text) > max_length:
        text = text[:max_length] + "... [TRUNCATED]"
        logger.info(f"Content truncated to {max_length} characters")
    
    return text


def _clean_url(url: str) -> str:
    """Clean and validate URL."""
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url


def scrape_tool(url: str, extract_text: bool = True, max_length: Optional[int] = None) -> str:
    """Enhanced website scraper with robust error handling and content extraction.
    
    Args:
        url: The URL to scrape
        extract_text: If True, extract clean text from HTML; if False, return raw HTML
        max_length: Maximum content length to return (default: MAX_CONTENT_LENGTH from env)
    
    Returns:
        Extracted content or error message starting with "TOOL_ERROR:"
    """
    if not url:
        return "TOOL_ERROR: empty URL provided"
    
    if max_length is None:
        max_length = MAX_CONTENT_LENGTH
    
    try:
        url = _clean_url(url)
        logger.info(f"Scraping URL: {url}")
        
        session = _create_session_with_retries(
            retries=SCRAPE_MAX_RETRIES,
            backoff_factor=1.5,
            timeout=SCRAPE_TIMEOUT
        )
        
        headers = {
            "User-Agent": SCRAPER_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
        
        response = session.get(
            url,
            headers=headers,
            timeout=SCRAPE_TIMEOUT,
            allow_redirects=True,
            verify=True  # SSL verification
        )
        
        # Log redirect chain if any
        if response.history:
            logger.info(f"Redirected: {url} -> {response.url}")
        
        # Check status code
        if response.status_code != 200:
            error_msg = f"TOOL_ERROR: HTTP {response.status_code} for {url}"
            logger.warning(error_msg)
            return error_msg
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type and 'text' not in content_type:
            error_msg = f"TOOL_ERROR: Non-HTML content type '{content_type}' from {url}"
            logger.warning(error_msg)
            return error_msg
        
        # Get content
        html = response.text
        
        if not html or len(html.strip()) == 0:
            error_msg = f"TOOL_ERROR: Empty response from {url}"
            logger.warning(error_msg)
            return error_msg
        
        logger.info(f"Successfully fetched {len(html)} bytes from {url}")
        
        # Extract or return raw
        if extract_text:
            content = _extract_text_from_html(html, max_length=max_length)
            logger.info(f"Extracted {len(content)} characters of text")
        else:
            content = html[:max_length] if max_length else html
        
        return content
        
    except requests.exceptions.Timeout as e:
        error_msg = f"TOOL_ERROR: Timeout after {SCRAPE_TIMEOUT}s connecting to {url}: {str(e)}"
        logger.error(error_msg)
        return error_msg
    
    except requests.exceptions.ConnectionError as e:
        error_msg = f"TOOL_ERROR: Connection failed to {url}: {str(e)}"
        logger.error(error_msg)
        return error_msg
    
    except requests.exceptions.RequestException as e:
        error_msg = f"TOOL_ERROR: Request failed for {url}: {str(e)}"
        logger.error(error_msg)
        return error_msg
    
    except Exception as e:
        error_msg = f"TOOL_ERROR: Unexpected error scraping {url}: {type(e).__name__}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg


@tool("Summarize Text")
def summarize_text_tool(text: str, max_sentences: int = 5) -> str:
    """Extract key sentences from text (simple extractive summary).
    
    Args:
        text: The text to summarize
        max_sentences: Maximum number of sentences to return
    
    Returns:
        Summarized text or error message
    """
    try:
        if not text or text.startswith("TOOL_ERROR:"):
            return text
        
        # Split into sentences (simple approach)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
        
        if not sentences:
            return "TOOL_ERROR: No sentences found in text"
        
        # Return first N sentences (could be improved with ranking)
        summary = '. '.join(sentences[:max_sentences]) + '.'
        logger.info(f"Summarized {len(sentences)} sentences to {max_sentences}")
        return summary
        
    except Exception as e:
        error_msg = f"TOOL_ERROR: Failed to summarize text: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool("Extract Links")
def extract_links_tool(url: str, filter_pattern: Optional[str] = None) -> str:
    """Extract all links from a webpage.
    
    Args:
        url: The URL to scrape
        filter_pattern: Optional regex pattern to filter links
    
    Returns:
        Newline-separated list of URLs or error message
    """
    try:
        html = scrape_tool(url, extract_text=False)
        
        if html.startswith("TOOL_ERROR:"):
            return html
        
        if not HAS_BS4:
            # Simple regex extraction
            links = re.findall(r'href=["\'](https?://[^"\']+)["\']', html)
        else:
            soup = BeautifulSoup(html, 'html.parser')
            links = [a.get('href') for a in soup.find_all('a', href=True)]
            links = [link for link in links if link.startswith(('http://', 'https://'))]
        
        if filter_pattern:
            pattern = re.compile(filter_pattern)
            links = [link for link in links if pattern.search(link)]
        
        links = list(set(links))  # Remove duplicates
        logger.info(f"Extracted {len(links)} links from {url}")
        
        return '\n'.join(links) if links else "No links found"
        
    except Exception as e:
        error_msg = f"TOOL_ERROR: Failed to extract links: {str(e)}"
        logger.error(error_msg)
        return error_msg


# Export summary of available tools
def get_tools_info() -> dict:
    """Return information about available tools."""
    return {
        "search_tool": "Available" if search_tool is not None else "Not available (missing SERPER_API_KEY or crewai_tools)",
        "scrape_tool": "Available with retries and HTML extraction",
        "summarize_text_tool": "Available - extractive summarization",
        "extract_links_tool": "Available - link extraction from pages",
        "logging": "Enabled" if ENABLE_TOOL_LOGGING else "Disabled",
        "log_file": str(log_file) if ENABLE_TOOL_LOGGING else None,
        "beautifulsoup": "Available" if HAS_BS4 else "Not available (will use regex fallback)"
    }

