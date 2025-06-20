import { useState, useRef, useEffect, useCallback } from 'react';
import { MagnifyingGlassIcon, MicrophoneIcon, XMarkIcon, TrashIcon, ClockIcon } from '@heroicons/react/24/outline';

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
  const [speechError, setSpeechError] = useState('');
  const [isSpeechSupported, setIsSpeechSupported] = useState(true);
  const inputRef = useRef(null);
  const resultsPerPage = 10;

  // Check backend status
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch('/api/health');
        if (response.ok) {
          setBackendStatus('connected');
        } else {
          setBackendStatus('error');
        }
      } catch {
        setBackendStatus('error');
      }
    };
    checkBackend();
  }, []);

  // Check speech recognition support
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setIsSpeechSupported(false);
      return;
    }

    recognitionRef.current = new SpeechRecognition();
    recognitionRef.current.continuous = false;
    recognitionRef.current.interimResults = false;
    recognitionRef.current.lang = 'en-US';

    recognitionRef.current.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setQuery(transcript);
      setIsListening(false);
      handleSearch({ preventDefault: () => {} }, transcript);
      setSpeechError('');
    };

    recognitionRef.current.onerror = (event) => {
      if (event.error === 'no-speech') {
        setSpeechError('No speech detected. Please try again.');
      } else if (event.error === 'not-allowed') {
        setSpeechError('Microphone access denied. Please enable permissions.');
      } else {
        setSpeechError('Voice recognition failed. Please try again.');
      }
      setIsListening(false);
    };

    recognitionRef.current.onend = () => {
      setIsListening(false);
    };

    return () => {
      if (recognitionRef.current) recognitionRef.current.stop();
    };
  }, []);

  const startListening = () => {
    setSpeechError('');
    if (!isSpeechSupported) {
      setSpeechError('Voice search not supported in your browser');
      return;
    }

    if (recognitionRef.current && !isListening) {
      try {
        recognitionRef.current.start();
        setIsListening(true);
      } catch (error) {
        setSpeechError('Failed to start microphone. Please check permissions.');
        setIsListening(false);
      }
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
        throw new Error('Invalid response from backend');
      }

      const data = await response.json();
      setResults(data.results || []);
      setTotalResults(data.total || 0);
      setTotalPages(Math.ceil((data.total || 0) / resultsPerPage));

      // Update search history
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

  // Delete individual history item
  const deleteHistoryItem = (itemToDelete) => {
    try {
      const history = JSON.parse(localStorage.getItem('searchHistory') || '[]');
      const newHistory = history.filter(item => item !== itemToDelete);
      localStorage.setItem('searchHistory', JSON.stringify(newHistory));
      
      // Update suggestions if showing history
      if (query.length < 3) {
        setSuggestions(newHistory.slice(0, 5));
      }
    } catch (error) {
      console.error('Error deleting history item:', error);
    }
  };

  // Clear all search history
  const clearAllHistory = () => {
    try {
      localStorage.setItem('searchHistory', JSON.stringify([]));
      setSuggestions([]);
    } catch (error) {
      console.error('Error clearing history:', error);
    }
  };

  const getSuggestions = useCallback(async (text) => {
    try {
      if (text.length > 0) {
        if (text.length < 3) {
          const history = JSON.parse(localStorage.getItem('searchHistory') || '[]');
          setSuggestions(Array.isArray(history) ? history.slice(0, 5) : []);
          return;
        }

        const response = await fetch(`/api/suggest?q=${encodeURIComponent(text)}`);
        if (!response.ok) {
          throw new Error('Invalid response from backend');
        }

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

  useEffect(() => {
    const timer = setTimeout(() => {
      getSuggestions(query);
    }, 300);
    return () => clearTimeout(timer);
  }, [query, getSuggestions]);

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

  const ResultsSkeleton = () => (
    <div className="mt-8 space-y-6 animate-pulse">
      {[...Array(5)].map((_, i) => (
        <div key={i} className="bg-white p-6 rounded-xl border border-gray-100 shadow-sm">
          <div className="h-4 bg-gray-200 rounded w-3/4 mb-4"></div>
          <div className="h-5 bg-gray-300 rounded w-full mb-3"></div>
          <div className="h-3 bg-gray-200 rounded w-5/6 mb-2"></div>
          <div className="h-3 bg-gray-200 rounded w-4/6 mb-2"></div>
          <div className="h-3 bg-gray-200 rounded w-2/4"></div>
        </div>
      ))}
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 flex flex-col items-center p-4 md:p-6">
      <div className="w-full max-w-4xl flex-grow flex flex-col">
        {/* Status indicator */}
        <div className={`text-right mb-2 text-xs font-medium ${backendStatus === 'connected' ? 'text-green-600' : backendStatus === 'error' ? 'text-red-600' : 'text-yellow-600'}`}>
          {backendStatus === 'connected' ? (
            <span className="flex items-center justify-end">
              <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
              Service available
            </span>
          ) : backendStatus === 'error' ? (
            <span className="flex items-center justify-end">
              <span className="w-2 h-2 bg-red-500 rounded-full mr-2"></span>
              Connection error - 
              <button onClick={() => window.location.reload()} className="ml-1 underline hover:text-red-700">
                Retry
              </button>
            </span>
          ) : (
            <span className="flex items-center justify-end">
              <span className="w-2 h-2 bg-yellow-500 rounded-full mr-2 animate-pulse"></span>
              Connecting...
            </span>
          )}
        </div>

        {/* Header */}
        <div className="text-center mb-8 mt-4">
          <h1 className="text-5xl md:text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-600 leading-tight">
            Nara<span className="font-light"> Power of Search</span>
          </h1>
          <p className="mt-3 text-gray-600 text-lg">Find what you need my real engine</p>
        </div>

        {/* Search form */}
        <form onSubmit={handleSearch} className="mb-6">
          <div className="relative flex items-center">
            <div className="absolute left-4 text-gray-400">
              <MagnifyingGlassIcon className="h-5 w-5" />
            </div>
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search anything or say your query..."
              className="w-full pl-12 pr-28 py-4 md:py-5 rounded-full border border-gray-200 shadow-lg focus:outline-none focus:ring-4 focus:ring-blue-200 focus:border-blue-400 text-base md:text-lg transition-all duration-200"
              autoComplete="off"
              aria-label="Search input"
            />
            {query && (
              <button
                type="button"
                onClick={clearSearch}
                className="absolute right-28 text-gray-400 hover:text-gray-600 transition-colors"
                aria-label="Clear search"
              >
                <XMarkIcon className="h-5 w-5" />
              </button>
            )}
            <div className="absolute right-2 flex space-x-2">
              <button
                type="button"
                onClick={startListening}
                disabled={!isSpeechSupported}
                className={`p-2 rounded-full hover:bg-gray-100 transition-colors ${isListening ? 'animate-pulse bg-red-100 text-red-500' : 'text-blue-500'} ${!isSpeechSupported ? 'opacity-50 cursor-not-allowed' : ''}`}
                aria-label="Voice search"
                title={!isSpeechSupported ? "Voice search not supported" : ""}
              >
                <MicrophoneIcon className="h-5 w-5" />
              </button>
              <button
                type="submit"
                disabled={backendStatus === 'error' || isLoading}
                className={`${backendStatus === 'error' || isLoading 
                  ? 'bg-gray-400 cursor-not-allowed' 
                  : 'bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 shadow-lg'
                } text-white px-5 py-2 md:px-6 md:py-2.5 rounded-full font-medium transition-all hover:shadow-xl transform hover:-translate-y-0.5 active:translate-y-0`}
              >
                {isLoading ? (
                  <span className="flex items-center">
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Searching...
                  </span>
                ) : 'Search'}
              </button>
            </div>
          </div>

          {/* Speech error message */}
          {speechError && (
            <div className="mt-2 ml-4 text-red-500 text-sm flex items-start animate-fadeIn">
              <XMarkIcon className="h-4 w-4 mr-1.5 mt-0.5 flex-shrink-0" />
              <span>{speechError}</span>
              {speechError.includes('try') && (
                <button 
                  onClick={startListening}
                  className="ml-2 text-blue-600 hover:underline font-medium"
                >
                  Try again
                </button>
              )}
            </div>
          )}

          {/* Browser support notice */}
          {!isSpeechSupported && (
            <div className="mt-2 ml-4 text-sm text-blue-600 flex items-center">
              <span>Voice search not supported in this browser. Try Chrome or Edge.</span>
            </div>
          )}

          {/* Suggestions dropdown */}
          {suggestions.length > 0 && (
            <div className="mt-2 bg-white rounded-xl shadow-xl border border-gray-200 max-h-60 overflow-y-auto z-10">
              {query.length < 3 && suggestions.length > 0 && (
                <div className="p-3 border-b border-gray-100 flex justify-between items-center bg-gray-50 sticky top-0">
                  <span className="text-xs font-medium text-gray-500 flex items-center">
                    <ClockIcon className="h-3.5 w-3.5 mr-1.5" />
                    RECENT SEARCHES
                  </span>
                  <button 
                    onClick={clearAllHistory}
                    className="text-xs text-red-500 hover:text-red-700 font-medium flex items-center transition-colors"
                  >
                    <TrashIcon className="h-3.5 w-3.5 mr-1" />
                    Clear all
                  </button>
                </div>
              )}
              {suggestions.map((suggestion, index) => (
                <div
                  key={index}
                  className="group p-3 hover:bg-blue-50 cursor-pointer border-b border-gray-100 last:border-b-0 flex items-center justify-between transition-colors"
                >
                  <div 
                    onClick={() => handleSuggestionClick(suggestion)}
                    className="flex-grow flex items-center"
                  >
                    {query.length < 3 ? (
                      <ClockIcon className="h-4 w-4 text-gray-400 mr-3 flex-shrink-0" />
                    ) : (
                      <MagnifyingGlassIcon className="h-4 w-4 text-blue-500 mr-3 flex-shrink-0" />
                    )}
                    <span className="truncate font-medium">{suggestion}</span>
                  </div>
                  {query.length < 3 && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteHistoryItem(suggestion);
                      }}
                      className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-500 p-1 transition-opacity"
                      aria-label="Delete this search history"
                    >
                      <XMarkIcon className="h-4 w-4" />
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
        </form>

        {/* Results section */}
        <div className="flex-grow mt-6">
          {isLoading ? (
            <ResultsSkeleton />
          ) : results.length > 0 ? (
            <>
              <div className="flex justify-between items-center mb-6">
                <p className="text-gray-600 text-sm font-medium">
                  About {totalResults.toLocaleString()} results
                </p>
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-500">
                    Page {currentPage} of {totalPages}
                  </span>
                </div>
              </div>
              
              <div className="space-y-6">
                {results.map((result, index) => (
                  <div 
                    key={index} 
                    className="bg-white p-6 rounded-xl border border-gray-100 shadow-sm hover:shadow-md transition-shadow duration-200"
                  >
                    <a href={result.url} className="block hover:underline" target="_blank" rel="noopener noreferrer">
                      <div className="flex items-center mb-3">
                        <div className="bg-gradient-to-r from-blue-500 to-indigo-500 w-4 h-4 rounded-full mr-2" />
                        <p className="text-xs text-gray-500 truncate">{result.url}</p>
                      </div>
                      <h3 className="text-xl font-semibold text-gray-800 mb-2 hover:text-blue-600 transition-colors">
                        {result.title}
                      </h3>
                    </a>
                    <p className="text-gray-600 mb-4">{result.description}</p>
                    {result.score && (
                      <div className="flex flex-wrap items-center gap-3">
                        <span className="text-xs bg-blue-50 text-blue-700 px-3 py-1.5 rounded-full font-medium">
                          Relevance: {Math.round(result.score * 100)}%
                        </span>
                        {result.lastUpdated && (
                          <span className="text-xs text-gray-500 font-medium">
                            Updated: {new Date(result.lastUpdated).toLocaleDateString()}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
              
              {/* Pagination */}
              {totalPages > 1 && (
                <div className="mt-10 flex justify-center space-x-3">
                  <button
                    onClick={() => handleSearch({ preventDefault: () => {} }, null, currentPage - 1)}
                    disabled={currentPage === 1}
                    className={`px-4 py-2 rounded-full font-medium transition-colors ${currentPage === 1 ? 'bg-gray-100 text-gray-400 cursor-not-allowed' : 'bg-gray-100 hover:bg-gray-200 text-gray-700'}`}
                  >
                    Previous
                  </button>
                  
                  {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                    let pageNum;
                    if (totalPages <= 5) {
                      pageNum = i + 1;
                    } else if (currentPage <= 3) {
                      pageNum = i + 1;
                    } else if (currentPage >= totalPages - 2) {
                      pageNum = totalPages - 4 + i;
                    } else {
                      pageNum = currentPage - 2 + i;
                    }

                    return (
                      <button
                        key={pageNum}
                        onClick={() => handleSearch({ preventDefault: () => {} }, null, pageNum)}
                        className={`px-4 py-2 rounded-full font-medium ${currentPage === pageNum ? 'bg-blue-600 text-white' : 'bg-gray-100 hover:bg-gray-200 text-gray-700'}`}
                      >
                        {pageNum}
                      </button>
                    );
                  })}
                  
                  <button
                    onClick={() => handleSearch({ preventDefault: () => {} }, null, currentPage + 1)}
                    disabled={currentPage === totalPages}
                    className={`px-4 py-2 rounded-full font-medium transition-colors ${currentPage === totalPages ? 'bg-gray-100 text-gray-400 cursor-not-allowed' : 'bg-gray-100 hover:bg-gray-200 text-gray-700'}`}
                  >
                    Next
                  </button>
                </div>
              )}
            </>
          ) : (
            query && !isLoading && (
              <div className="text-center py-12 md:py-16 flex-grow flex flex-col justify-center items-center">
                <div className="bg-gradient-to-br from-blue-50 to-indigo-50 inline-block p-6 rounded-2xl mb-6">
                  <MagnifyingGlassIcon className="h-16 w-16 text-blue-400 mx-auto" />
                </div>
                <h2 className="text-2xl font-bold text-gray-800 mb-3">No results found</h2>
                <p className="text-gray-600 max-w-md mx-auto mb-8">
                  Try different keywords or check your spelling
                </p>
                <div className="flex justify-center gap-4 flex-wrap">
                  {isSpeechSupported && (
                    <button 
                      onClick={startListening}
                      className="flex items-center justify-center bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-6 py-3 rounded-full hover:from-blue-700 hover:to-indigo-700 transition-all shadow-md hover:shadow-lg"
                    >
                      <MicrophoneIcon className="h-5 w-5 mr-2" />
                      Try Voice Search
                    </button>
                  )}
                  <button 
                    onClick={clearSearch}
                    className="flex items-center justify-center bg-white text-gray-800 px-6 py-3 rounded-full hover:bg-gray-50 transition-colors shadow-md border border-gray-200"
                  >
                    <XMarkIcon className="h-5 w-5 mr-2" />
                    Clear Search
                  </button>
                </div>
              </div>
            )
          )}
        </div>

        {/* Footer */}
        <footer className="mt-12 py-6 text-center text-gray-500 text-sm">
          <div className="flex flex-wrap justify-center items-center gap-4 mb-3">
            <a href="#" className="hover:text-blue-600 transition-colors">Privacy</a>
            <a href="#" className="hover:text-blue-600 transition-colors">Terms</a>
            <a href="#" className="hover:text-blue-600 transition-colors">Settings</a>
            <a href="#" className="hover:text-blue-600 transition-colors">Help</a>
          </div>
          <p>© {new Date().getFullYear()} SearchEngine • Powered by advanced algorithms</p>
        </footer>
      </div>
    </div>
  );
};

export default SearchEngine;