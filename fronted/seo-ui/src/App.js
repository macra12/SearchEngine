import { useState, useRef, useEffect, useCallback } from 'react';
import { MagnifyingGlassIcon, MicrophoneIcon, XMarkIcon } from '@heroicons/react/24/outline';

const SearchEngine = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [backendStatus, setBackendStatus] = useState('unknown');
  const recognitionRef = useRef(null);
  const [suggestions, setSuggestions] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalResults, setTotalResults] = useState(0);
  const inputRef = useRef(null);
  const resultsPerPage = 10;

  // Check backend connection on startup
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch('/api/health');
        if (response.ok) {
          setBackendStatus('connected');
        } else {
          setBackendStatus('error');
        }
      } catch (error) {
        setBackendStatus('error');
      }
    };
    
    checkBackend();
  }, []);

  // Initialize speech recognition
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setQuery(transcript);
        setIsListening(false);
        handleSearch({ preventDefault: () => {} }, transcript);
      };

      recognitionRef.current.onerror = (event) => {
        console.error('Speech recognition error', event.error);
        setIsListening(false);
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  const startListening = () => {
    if (recognitionRef.current && !isListening) {
      recognitionRef.current.start();
      setIsListening(true);
    }
  };

  const handleSearch = async (e, voiceQuery = null, page = 1) => {
    e.preventDefault();
    const searchQuery = voiceQuery || query;
    
    if (!searchQuery.trim()) return;
    
    setIsLoading(true);
    setCurrentPage(page);
    try {
      const response = await fetch(
        `/api/search?q=${encodeURIComponent(searchQuery)}&page=${page}&limit=${resultsPerPage}`
      );
      
      if (!response.ok) {
        throw new Error(`Backend error: ${response.status}`);
      }
      
      const data = await response.json();
      setResults(data.results || []);
      setTotalResults(data.total || 0);
      setTotalPages(Math.ceil((data.total || 0) / resultsPerPage));
      
      // Save to search history
      try {
        const history = JSON.parse(localStorage.getItem('searchHistory') || '[]');
        const newHistory = [searchQuery, ...history.filter(item => item !== searchQuery)].slice(0, 5);
        localStorage.setItem('searchHistory', JSON.stringify(newHistory));
      } catch (storageError) {
        console.error('LocalStorage error:', storageError);
      }
    } catch (error) {
      console.error('Search error:', error);
      setResults([]);
      setBackendStatus('error');
    } finally {
      setIsLoading(false);
    }
  };

  // Get search suggestions with debounce
  const getSuggestions = useCallback(async (text) => {
    try {
      if (text.length > 0) {
        // Show history for short queries
        if (text.length < 3) {
          const history = JSON.parse(localStorage.getItem('searchHistory') || '[]');
          setSuggestions(Array.isArray(history) ? history.slice(0, 5) : []);
          return;
        }
        
        const response = await fetch(`/api/suggest?q=${encodeURIComponent(text)}`);
        
        if (!response.ok) return;
        
        const data = await response.json();
        setSuggestions(data.suggestions || []);
      } else {
        const history = JSON.parse(localStorage.getItem('searchHistory') || '[]');
        setSuggestions(Array.isArray(history) ? history.slice(0, 5) : []);
      }
    } catch (error) {
      console.error('Suggestion error:', error);
    }
  }, []);

  // Debounced suggestion fetching
  useEffect(() => {
    const timer = setTimeout(() => {
      getSuggestions(query);
    }, 300);

    return () => clearTimeout(timer);
  }, [query, getSuggestions]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        setSuggestions([]);
      }
      if (e.key === '/' && document.activeElement !== inputRef.current) {
        e.preventDefault();
        inputRef.current.focus();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const clearSearch = () => {
    setQuery('');
    setSuggestions([]);
    inputRef.current.focus();
  };

  const handleSuggestionClick = (suggestion) => {
    setQuery(suggestion);
    handleSearch({ preventDefault: () => {} }, suggestion);
  };

  // Skeleton loader for results
  const ResultsSkeleton = () => (
    <div className="mt-8 space-y-6">
      {[...Array(5)].map((_, i) => (
        <div key={i} className="bg-white p-4 rounded-lg border border-gray-100">
          <div className="animate-pulse flex flex-col">
            <div className="h-4 bg-gray-200 rounded w-3/4 mb-3"></div>
            <div className="h-5 bg-gray-300 rounded w-full mb-2"></div>
            <div className="h-3 bg-gray-200 rounded w-5/6 mb-1"></div>
            <div className="h-3 bg-gray-200 rounded w-4/6 mb-1"></div>
            <div className="h-3 bg-gray-200 rounded w-2/4"></div>
          </div>
        </div>
      ))}
    </div>
  );

  return (
    <div className="min-h-screen bg-white flex flex-col items-center p-4">
      <div className="w-full max-w-3xl">
        {/* Backend status indicator - moved to top right */}
        <div className={`text-right mb-2 text-xs ${
          backendStatus === 'connected' ? 'text-green-600' : 
          backendStatus === 'error' ? 'text-red-600' : 'text-yellow-600'
        }`}>
          {backendStatus === 'connected' ? 'Connected' : 
           backendStatus === 'error' ? (
             <span>
               Connection error - 
               <button 
                 onClick={() => window.location.reload()} 
                 className="ml-1 underline hover:text-red-700"
               >
                 Retry
               </button>
             </span>
           ) : 
           'Connecting...'}
        </div>

        {/* Logo/Title - Centered with larger text */}
        <div className="text-center mb-10">
          <h1 className="text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-600">
            Search<span className="font-light">Engine</span>
          </h1>
          <p className="mt-3 text-gray-500">Search the web using TF-IDF and voice recognition</p>
        </div>

        {/* Search Box - Larger and more prominent */}
        <form onSubmit={handleSearch} className="mb-4">
          <div className="relative flex items-center">
            <div className="absolute left-4 text-gray-400">
              <MagnifyingGlassIcon className="h-5 w-5" />
            </div>
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search or speak your query..."
              className="w-full pl-12 pr-24 py-5 rounded-full border border-gray-300 shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-lg transition-all"
              autoComplete="off"
            />
            {query && (
              <button
                type="button"
                onClick={clearSearch}
                className="absolute right-28 text-gray-400 hover:text-gray-600"
                aria-label="Clear search"
              >
                <XMarkIcon className="h-5 w-5" />
              </button>
            )}
            <div className="absolute right-2 flex space-x-2">
              <button
                type="button"
                onClick={startListening}
                className={`p-2 rounded-full hover:bg-gray-100 transition-colors ${isListening ? 'animate-pulse bg-red-100' : ''}`}
                aria-label="Voice search"
              >
                <MicrophoneIcon className={`h-5 w-5 ${isListening ? 'text-red-500' : 'text-blue-500'}`} />
              </button>
              <button
                type="submit"
                disabled={backendStatus === 'error' || isLoading}
                className={`${
                  backendStatus === 'error' || isLoading 
                    ? 'bg-gray-400 cursor-not-allowed' 
                    : 'bg-blue-600 hover:bg-blue-700'
                } text-white px-6 py-2.5 rounded-full font-medium transition-colors shadow-md hover:shadow-lg`}
              >
                {isLoading ? 'Searching...' : 'Search'}
              </button>
            </div>
          </div>
          
          {/* Suggestions - Styled like Google */}
          {suggestions.length > 0 && (
            <div className="mt-1 bg-white rounded-lg shadow-lg border border-gray-200 max-h-60 overflow-y-auto z-10">
              {suggestions.map((suggestion, index) => (
                <div
                  key={index}
                  onClick={() => handleSuggestionClick(suggestion)}
                  className="p-3 hover:bg-gray-100 cursor-pointer border-b border-gray-100 last:border-b-0 flex items-center"
                >
                  <MagnifyingGlassIcon className="h-4 w-4 text-gray-400 mr-3 flex-shrink-0" />
                  <span className="truncate">{suggestion}</span>
                </div>
              ))}
            </div>
          )}
        </form>

        {/* Results Section */}
        <div className="mt-8">
          {isLoading ? (
            <ResultsSkeleton />
          ) : results.length > 0 ? (
            <>
              <p className="text-gray-600 mb-6 text-sm">
                About {totalResults.toLocaleString()} results ({currentPage} of {totalPages})
              </p>
              
              <div className="space-y-8">
                {results.map((result, index) => (
                  <div key={index} className="bg-white p-0">
                    <a href={result.url} className="block hover:underline" target="_blank" rel="noopener noreferrer">
                      <div className="flex items-center mb-1">
                        {/* Favicon placeholder */}
                        <div className="bg-gray-200 border-2 border-dashed rounded w-4 h-4 mr-2" />
                        <p className="text-xs text-gray-700 truncate">{result.url}</p>
                      </div>
                      <h3 className="text-xl font-medium text-blue-600 mt-1 mb-2">{result.title}</h3>
                    </a>
                    <p className="text-gray-600 text-sm">{result.description}</p>
                    {result.score && (
                      <div className="flex items-center justify-between mt-3">
                        <span className="text-xs bg-blue-50 text-blue-700 px-2 py-1 rounded">
                          Relevance: {Math.round(result.score * 100)}%
                        </span>
                        <span className="text-xs text-gray-500">
                          Last updated: {result.lastUpdated || 'Unknown'}
                        </span>
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {/* Pagination - Centered and styled */}
              {totalPages > 1 && (
                <div className="mt-12 flex justify-center space-x-2">
                  <button
                    onClick={() => handleSearch({ preventDefault: () => {} }, query, currentPage - 1)}
                    disabled={currentPage === 1}
                    className={`px-4 py-2 rounded-md text-sm ${
                      currentPage === 1 
                        ? 'bg-gray-100 text-gray-400 cursor-not-allowed' 
                        : 'bg-white text-blue-600 hover:bg-blue-50'
                    } border border-gray-200`}
                  >
                    Previous
                  </button>
                  
                  {Array.from({ length: Math.min(totalPages, 5) }, (_, i) => {
                    const pageNum = i + 1;
                    return (
                      <button
                        key={i}
                        onClick={() => handleSearch({ preventDefault: () => {} }, query, pageNum)}
                        className={`px-4 py-2 rounded-md text-sm ${
                          currentPage === pageNum
                            ? 'bg-blue-600 text-white'
                            : 'bg-white text-blue-600 hover:bg-blue-50'
                        } border border-gray-200`}
                      >
                        {pageNum}
                      </button>
                    );
                  })}
                  
                  {totalPages > 5 && (
                    <span className="px-4 py-2 text-gray-500 text-sm">...</span>
                  )}
                  
                  <button
                    onClick={() => handleSearch({ preventDefault: () => {} }, query, currentPage + 1)}
                    disabled={currentPage === totalPages}
                    className={`px-4 py-2 rounded-md text-sm ${
                      currentPage === totalPages
                        ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                        : 'bg-white text-blue-600 hover:bg-blue-50'
                    } border border-gray-200`}
                  >
                    Next
                  </button>
                </div>
              )}
            </>
          ) : (
            query && !isLoading && (
              <div className="text-center py-16">
                <div className="bg-gray-100 inline-block p-5 rounded-full mb-6">
                  <MagnifyingGlassIcon className="h-14 w-14 text-gray-400 mx-auto" />
                </div>
                <p className="text-2xl font-medium text-gray-800 mb-3">No results found</p>
                <p className="text-gray-600 max-w-md mx-auto mb-8">
                  Try different keywords or voice search. Make sure all words are spelled correctly.
                </p>
                <div className="flex justify-center gap-4">
                  <button 
                    onClick={() => startListening()}
                    className="flex items-center justify-center bg-blue-600 text-white px-6 py-3 rounded-full hover:bg-blue-700 transition-colors shadow-md"
                  >
                    <MicrophoneIcon className="h-5 w-5 mr-2" />
                    Try Voice Search
                  </button>
                  <button 
                    onClick={() => setQuery('')}
                    className="flex items-center justify-center bg-gray-100 text-gray-800 px-6 py-3 rounded-full hover:bg-gray-200 transition-colors shadow-md"
                  >
                    <XMarkIcon className="h-5 w-5 mr-2" />
                    Clear Search
                  </button>
                </div>
              </div>
            )
          )}
        </div>

        {/* Footer - Simplified and fixed to bottom */}
        <footer className="mt-auto w-full py-6 text-center text-gray-500 text-sm">
          <div className="flex flex-wrap justify-center items-center gap-4 mb-2">
            <span>Privacy</span>
            <span>Terms</span>
            <span>Settings</span>
            <span>Help</span>
          </div>
          <p>© {new Date().getFullYear()} SearchEngine. Using advanced TF-IDF algorithms.</p>
        </footer>
      </div>
    </div>
  );
};

export default SearchEngine;