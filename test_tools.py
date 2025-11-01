"""Quick test script to verify enhanced tools functionality."""

import sys
from tools import scrape_tool, get_tools_info, extract_links_tool, summarize_text_tool

def test_tools():
    """Test all tools with safe public URLs."""
    print("="*60)
    print("TESTING ENHANCED TOOLS")
    print("="*60)
    
    # Show available tools
    print("\n1. Tools Info:")
    info = get_tools_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Test scraping with a reliable site
    print("\n2. Testing scrape_tool with example.com...")
    result = scrape_tool("https://example.com", extract_text=True, max_length=500)
    if result.startswith("TOOL_ERROR:"):
        print(f"   ❌ Error: {result}")
    else:
        print(f"   ✓ Success: {len(result)} chars extracted")
        print(f"   Preview: {result[:100]}...")
    
    # Test link extraction
    print("\n3. Testing extract_links_tool...")
    links = extract_links_tool("https://example.com")
    if links.startswith("TOOL_ERROR:"):
        print(f"   ❌ Error: {links}")
    else:
        link_list = links.split('\n')
        print(f"   ✓ Success: {len(link_list)} links found")
        if link_list:
            print(f"   First link: {link_list[0]}")
    
    # Test summarization
    print("\n4. Testing summarize_text_tool...")
    sample_text = """
    Artificial intelligence is transforming modern healthcare in remarkable ways. 
    Machine learning algorithms can now detect diseases earlier than traditional methods. 
    AI-powered diagnostic tools are helping doctors make more accurate decisions. 
    Personalized treatment plans are being developed using patient data analysis. 
    The future of medicine will likely involve significant AI integration.
    However, ethical considerations and data privacy remain important concerns.
    """
    summary = summarize_text_tool(sample_text, max_sentences=2)
    if summary.startswith("TOOL_ERROR:"):
        print(f"   ❌ Error: {summary}")
    else:
        print(f"   ✓ Success: {len(summary)} chars")
        print(f"   Summary: {summary}")
    
    # Test error handling with bad URL
    print("\n5. Testing error handling with invalid URL...")
    error_result = scrape_tool("https://this-domain-definitely-does-not-exist-123456.com")
    if error_result.startswith("TOOL_ERROR:"):
        print(f"   ✓ Properly handled error: {error_result[:80]}...")
    else:
        print(f"   ❌ Unexpected success")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    try:
        test_tools()
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        sys.exit(1)
