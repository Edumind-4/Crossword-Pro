/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useMemo, useCallback, useEffect } from 'react';
import { GoogleGenAI, Type } from "@google/genai";
import { motion, AnimatePresence } from "motion/react";
import { doc, getDoc, setDoc, serverTimestamp } from 'firebase/firestore';
import { nanoid } from 'nanoid';
import { db } from './lib/firebase';
import { 
  Puzzle, 
  Sparkles, 
  Type as TypeIcon, 
  Printer, 
  Key, 
  RefreshCcw, 
  AlertCircle,
  Hash,
  List as ListIcon,
  ChevronRight,
  Loader2,
  Share2,
  Check,
  Link as LinkIcon
} from "lucide-react";

// --- Types ---

interface WordData {
  word: string;
  clue: string;
}

interface PlacedWord extends WordData {
  x: number;
  y: number;
  isAcross: boolean;
  num?: number;
}

interface GridResult {
  grid: (string | null)[][];
  placedWords: PlacedWord[];
  failedWords: string[];
  rows: number;
  cols: number;
}

// --- Utils ---

const GRID_SIZE = 40;

const buildCrossword = (wordData: WordData[]): GridResult => {
  // Sort longest to shortest
  const sorted = [...wordData].sort((a, b) => b.word.length - a.word.length);
  
  let grid = Array(GRID_SIZE).fill(null).map(() => Array(GRID_SIZE).fill(null));
  let placedWords: PlacedWord[] = [];
  let failedWords: string[] = [];

  const canPlace = (word: string, startX: number, startY: number, isAcross: boolean) => {
    if (isAcross && startX + word.length > GRID_SIZE) return false;
    if (!isAcross && startY + word.length > GRID_SIZE) return false;
    if (startX < 0 || startY < 0) return false;

    let intersections = 0;

    for (let i = 0; i < word.length; i++) {
      let cx = startX + (isAcross ? i : 0);
      let cy = startY + (!isAcross ? i : 0);
      let char = word[i];

      if (grid[cy][cx] !== null && grid[cy][cx] !== char) return false;
      if (grid[cy][cx] === char) intersections++;

      // Check parallel collisions
      if (grid[cy][cx] === null) {
        if (isAcross) {
          if (cy > 0 && grid[cy - 1][cx] !== null) return false;
          if (cy < GRID_SIZE - 1 && grid[cy + 1][cx] !== null) return false;
        } else {
          if (cx > 0 && grid[cy][cx - 1] !== null) return false;
          if (cx < GRID_SIZE - 1 && grid[cy][cx + 1] !== null) return false;
        }
      }
    }

    // Check front and back bounds
    if (isAcross) {
      if (startX > 0 && grid[startY][startX - 1] !== null) return false;
      if (startX + word.length < GRID_SIZE && grid[startY][startX + word.length] !== null) return false;
    } else {
      if (startY > 0 && grid[startY - 1][startX] !== null) return false;
      if (startY + word.length < GRID_SIZE && grid[startY + word.length][startX] !== null) return false;
    }

    return intersections > 0;
  };

  // Place first word
  const first = sorted[0];
  if (first) {
    const startX = Math.floor(GRID_SIZE / 2) - Math.floor(first.word.length / 2);
    const startY = Math.floor(GRID_SIZE / 2);
    for (let i = 0; i < first.word.length; i++) {
      grid[startY][startX + i] = first.word[i];
    }
    placedWords.push({ ...first, x: startX, y: startY, isAcross: true });

    // Place remaining
    for (let w = 1; w < sorted.length; w++) {
      let currentWord = sorted[w].word;
      let placed = false;

      for (let i = 0; i < placedWords.length && !placed; i++) {
        let target = placedWords[i];
        for (let c1 = 0; c1 < currentWord.length && !placed; c1++) {
          for (let c2 = 0; c2 < target.word.length && !placed; c2++) {
            if (currentWord[c1] === target.word[c2]) {
              let pX = target.isAcross ? target.x + c2 : target.x - c1;
              let pY = target.isAcross ? target.y - c1 : target.y + c2;
              let pAcross = !target.isAcross;

              if (canPlace(currentWord, pX, pY, pAcross)) {
                for (let idx = 0; idx < currentWord.length; idx++) {
                  grid[pY + (!pAcross ? idx : 0)][pX + (pAcross ? idx : 0)] = currentWord[idx];
                }
                placedWords.push({ ...sorted[w], x: pX, y: pY, isAcross: pAcross });
                placed = true;
              }
            }
          }
        }
      }
      if (!placed) failedWords.push(sorted[w].word);
    }
  }

  // Crop
  let minX = GRID_SIZE, maxX = 0, minY = GRID_SIZE, maxY = 0;
  if (placedWords.length === 0) return { grid: [], placedWords: [], failedWords: [], rows: 0, cols: 0 };
  
  placedWords.forEach(pw => {
    minX = Math.min(minX, pw.x);
    minY = Math.min(minY, pw.y);
    maxX = Math.max(maxX, pw.isAcross ? pw.x + pw.word.length - 1 : pw.x);
    maxY = Math.max(maxY, !pw.isAcross ? pw.y + pw.word.length - 1 : pw.y);
  });

  const rows = maxY - minY + 1;
  const cols = maxX - minX + 1;
  const croppedGrid = Array(rows).fill(null).map(() => Array(cols).fill(null));
  
  placedWords.forEach(pw => {
    pw.x -= minX;
    pw.y -= minY;
  });

  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      croppedGrid[y][x] = grid[y + minY][x + minX];
    }
  }

  // Numbering
  placedWords.sort((a, b) => (a.y === b.y ? a.x - b.x : a.y - b.y));
  let currentNum = 1;
  const numberMap: Record<string, number> = {};
  
  placedWords.forEach(pw => {
    const key = `${pw.x},${pw.y}`;
    if (numberMap[key]) {
      pw.num = numberMap[key];
    } else {
      pw.num = currentNum++;
      numberMap[key] = pw.num;
    }
  });

  return { grid: croppedGrid, placedWords, failedWords, rows, cols };
};

// --- Components ---

export default function App() {
  const [mode, setMode] = useState<'manual' | 'ai'>('manual');
  const [manualText, setManualText] = useState("ALGORITHM : A set of instructions for a computer\nBROWSER : Software used to view web pages\nFIREWALL : Security system to block unauthorized access\nMALWARE : Malicious software designed to cause harm\nPHISHING : Fake emails trying to steal your passwords");
  const [aiText, setAiText] = useState("");
  const [aiCount, setAiCount] = useState(10);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [gridResult, setGridResult] = useState<GridResult | null>(null);
  const [showKey, setShowKey] = useState(false);
  const [userInput, setUserInput] = useState<Record<string, string>>({});
  const [focusedCell, setFocusedCell] = useState<{ x: number, y: number, dir: 'across' | 'down' } | null>(null);
  const [shareSuccess, setShareSuccess] = useState(false);

  const ai = useMemo(() => new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || '' }), []);

  // Handle Loading Shared Puzzle
  useEffect(() => {
    const loadPuzzle = async () => {
      const params = new URLSearchParams(window.location.search);
      const shortId = params.get('p');
      const puzzleData = params.get('puzzle');

      let data: any = null;

      if (shortId) {
        setLoading(true);
        try {
          const docSnap = await getDoc(doc(db, 'puzzles', shortId));
          if (docSnap.exists()) {
            data = docSnap.data();
          } else {
            setError("Shared puzzle not found. It may have expired or the link is incorrect.");
          }
        } catch (e) {
          console.error("Error fetching shared puzzle:", e);
          setError("Failed to connect to the database. Trying fallback...");
        } finally {
          setLoading(false);
        }
      }

      // Fallback to long puzzle data
      if (!data && puzzleData) {
        try {
          const decodedStr = decodeURIComponent(escape(atob(puzzleData)));
          const decoded = JSON.parse(decodedStr);
          data = {
            placedWords: decoded.p.map((p: any) => ({
              word: p[0],
              clue: p[1],
              x: p[2],
              y: p[3],
              isAcross: p[4]
            })),
            rows: decoded.r,
            cols: decoded.c
          };
        } catch (e) {
          console.error("Failed to load fallback puzzle data", e);
          setError("The shared link appears to be invalid or corrupted.");
        }
      }

      if (data) {
        const { placedWords, rows, cols } = data;
        const grid = Array(rows).fill(null).map(() => Array(cols).fill(null));
        placedWords.forEach((pw: any) => {
          for (let i = 0; i < pw.word.length; i++) {
            const cx = pw.x + (pw.isAcross ? i : 0);
            const cy = pw.y + (!pw.isAcross ? i : 0);
            grid[cy][cx] = pw.word[i];
          }
        });

        const sortedWords = [...placedWords].sort((a: any, b: any) => (a.y === b.y ? a.x - b.x : a.y - b.y));
        let currentNum = 1;
        const numberMap: Record<string, number> = {};
        sortedWords.forEach((pw: any) => {
          const key = `${pw.x},${pw.y}`;
          if (numberMap[key]) {
            pw.num = numberMap[key];
          } else {
            pw.num = currentNum++;
            numberMap[key] = pw.num;
          }
        });

        setGridResult({
          grid,
          placedWords: sortedWords,
          failedWords: [],
          rows,
          cols
        });
      }
    };

    loadPuzzle();
  }, []);

  // Programmatic Focus Management
  useEffect(() => {
    if (focusedCell) {
      const el = document.querySelector(`[data-coord="${focusedCell.x},${focusedCell.y}"]`) as HTMLInputElement;
      if (el) el.focus();
    }
  }, [focusedCell]);

  const handleManualGenerate = useCallback(() => {
    const lines = manualText.split('\n');
    const words: WordData[] = [];
    lines.forEach(line => {
      if (line.includes(':')) {
        const parts = line.split(':');
        const word = parts[0].trim().toUpperCase().replace(/[^A-Z]/g, '');
        const clue = parts[1].trim();
        if (word.length >= 2 && clue.length > 0) {
          words.push({ word, clue });
        }
      }
    });

    if (words.length < 2) return;
    setGridResult(buildCrossword(words));
    setUserInput({});
    setFocusedCell(null);
  }, [manualText]);

  const moveFocus = useCallback((x: number, y: number, dir: 'across' | 'down', delta: number) => {
    if (!gridResult) return;
    
    let nextX = x + (dir === 'across' ? delta : 0);
    let nextY = y + (dir === 'down' ? delta : 0);

    // Skip black cells
    while (
      nextX >= 0 && nextX < gridResult.cols && 
      nextY >= 0 && nextY < gridResult.rows && 
      gridResult.grid[nextY][nextX] === null
    ) {
      nextX += (dir === 'across' ? delta : 0);
      nextY += (dir === 'down' ? delta : 0);
    }

    if (nextX >= 0 && nextX < gridResult.cols && nextY >= 0 && nextY < gridResult.rows) {
      setFocusedCell({ x: nextX, y: nextY, dir });
    }
  }, [gridResult]);

  const handleShare = useCallback(async () => {
    if (!gridResult) return;
    
    setLoading(true);
    try {
      const shortId = nanoid(8);
      const puzzleRecord = {
        placedWords: gridResult.placedWords.map(pw => ({
          word: pw.word,
          clue: pw.clue,
          x: pw.x,
          y: pw.y,
          isAcross: pw.isAcross
        })),
        rows: gridResult.rows,
        cols: gridResult.cols,
        createdAt: serverTimestamp()
      };
      
      await setDoc(doc(db, 'puzzles', shortId), puzzleRecord);
      
      const url = new URL(window.location.origin + window.location.pathname);
      url.searchParams.set('p', shortId);
      
      navigator.clipboard.writeText(url.toString());
      setShareSuccess(true);
      setTimeout(() => setShareSuccess(false), 2000);
    } catch (e) {
      console.error("Failed to share puzzle to DB, falling back to long link", e);
      
      // Fallback to long link
      try {
        const data = {
          p: gridResult.placedWords.map(pw => [pw.word, pw.clue, pw.x, pw.y, pw.isAcross]),
          r: gridResult.rows,
          c: gridResult.cols
        };
        const serialized = btoa(unescape(encodeURIComponent(JSON.stringify(data))));
        const url = new URL(window.location.origin + window.location.pathname);
        url.searchParams.set('puzzle', serialized);
        
        navigator.clipboard.writeText(url.toString());
        setShareSuccess(true);
        setTimeout(() => setShareSuccess(false), 2000);
      } catch (err) {
        console.error("Total share failure", err);
        alert("Could not generate share link.");
      }
    } finally {
      setLoading(false);
    }
  }, [gridResult]);

  const handleCellClick = (x: number, y: number) => {
    if (!gridResult) return;

    const hasAcross = gridResult.placedWords.some(pw => 
      pw.isAcross && x >= pw.x && x < pw.x + pw.word.length && y === pw.y
    );
    const hasDown = gridResult.placedWords.some(pw => 
      !pw.isAcross && y >= pw.y && y < pw.y + pw.word.length && x === pw.x
    );

    if (focusedCell?.x === x && focusedCell?.y === y) {
      if (hasAcross && hasDown) {
        setFocusedCell({ x, y, dir: focusedCell.dir === 'across' ? 'down' : 'across' });
      }
    } else {
      if (hasAcross) {
        setFocusedCell({ x, y, dir: 'across' });
      } else if (hasDown) {
        setFocusedCell({ x, y, dir: 'down' });
      }
    }
  };

  const handleInputChange = (x: number, y: number, val: string) => {
    const char = val.slice(-1).toUpperCase();
    if (!char.match(/[A-Z]/) && char !== "") return;
    
    setUserInput(prev => ({ ...prev, [`${x},${y}`]: char }));
    
    if (char && focusedCell) {
      moveFocus(x, y, focusedCell.dir, 1);
    }
  };

  const handleKeyDown = (e: import('react').KeyboardEvent, x: number, y: number) => {
    if (e.key === 'Backspace' && !userInput[`${x},${y}`]) {
      e.preventDefault();
      moveFocus(x, y, focusedCell?.dir || 'across', -1);
    } else if (e.key === 'ArrowRight') {
      moveFocus(x, y, 'across', 1);
    } else if (e.key === 'ArrowLeft') {
      moveFocus(x, y, 'across', -1);
    } else if (e.key === 'ArrowUp') {
      moveFocus(x, y, 'down', -1);
    } else if (e.key === 'ArrowDown') {
      moveFocus(x, y, 'down', 1);
    } else if (e.key === 'Tab') {
      // Logic for next clue could go here, but default tab is okay for now
    }
  };

  const handleRevealWord = () => {
    if (!activeClue || !gridResult) return;
    const { x, y, word, isAcross } = activeClue;
    const newInputs = { ...userInput };
    for (let i = 0; i < word.length; i++) {
      const cx = x + (isAcross ? i : 0);
      const cy = y + (!isAcross ? i : 0);
      newInputs[`${cx},${cy}`] = word[i];
    }
    setUserInput(newInputs);
  };

  const isCellInActiveWord = (x: number, y: number) => {
    if (!focusedCell || !gridResult) return false;
    const activeWord = gridResult.placedWords.find(pw => 
      pw.isAcross === (focusedCell.dir === 'across') &&
      x >= pw.x && x < pw.x + (pw.isAcross ? pw.word.length : 1) &&
      y >= pw.y && y < pw.y + (!pw.isAcross ? pw.word.length : 1) &&
      focusedCell.x >= pw.x && focusedCell.x < pw.x + (pw.isAcross ? pw.word.length : 1) &&
      focusedCell.y >= pw.y && focusedCell.y < pw.y + (!pw.isAcross ? pw.word.length : 1)
    );
    return !!activeWord;
  };

  const handleAiExtract = async () => {
    if (!aiText.trim()) return;
    setLoading(true);
    setError(null);

    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      setError("Gemini API Key is missing. If you are deploying this yourself, please ensure GEMINI_API_KEY is set in your environment variables. In AI Studio, ensure you have set it in the Secrets panel.");
      setLoading(false);
      return;
    }

    try {
      const response = await ai.models.generateContent({
        model: "gemini-3.1-flash-lite-preview",
        contents: `Analyze the following text. Extract exactly ${aiCount} challenging or thematic vocabulary words from it. 
        For each word, write a clear, 1-sentence clue suitable for a crossword puzzle.
        Output as a JSON array of objects with keys "word" and "clue". Letters only for words.
        
        TEXT: ${aiText}`,
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                word: { type: Type.STRING },
                clue: { type: Type.STRING }
              },
              required: ["word", "clue"]
            }
          }
        }
      });

      const extracted: WordData[] = JSON.parse(response.text || "[]");
      setManualText(extracted.map(item => `${item.word.toUpperCase()} : ${item.clue}`).join('\n'));
      setMode('manual');
      const res = buildCrossword(extracted.map(item => ({...item, word: item.word.toUpperCase().replace(/[^A-Z]/g, '')})));
      setGridResult(res);
      setUserInput({});
      setFocusedCell(null);
    } catch (err: any) {
      console.error(err);
      setError(err?.message || "An unexpected error occurred during AI extraction.");
    } finally {
      setLoading(false);
    }
  };

  const acrossClues = useMemo(() => gridResult?.placedWords.filter(pw => pw.isAcross).sort((a,b) => (a.num || 0) - (b.num || 0)) || [], [gridResult]);
  const downClues = useMemo(() => gridResult?.placedWords.filter(pw => !pw.isAcross).sort((a,b) => (a.num || 0) - (b.num || 0)) || [], [gridResult]);

  const activeClue = useMemo(() => {
    if (!focusedCell || !gridResult) return null;
    return gridResult.placedWords.find(pw => 
      pw.isAcross === (focusedCell.dir === 'across') &&
      focusedCell.x >= pw.x && focusedCell.x < pw.x + (pw.isAcross ? pw.word.length : 1) &&
      focusedCell.y >= pw.y && focusedCell.y < pw.y + (!pw.isAcross ? pw.word.length : 1)
    );
  }, [focusedCell, gridResult]);

  return (
    <div className="min-h-screen bg-[#F8FAFC] text-slate-900 font-sans p-4 md:p-8">
      <div className="max-w-5xl mx-auto">
        
        {/* Header */}
          <header className="text-center mb-8 md:mb-12">
            <motion.div 
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="inline-flex items-center justify-center w-12 h-12 md:w-16 md:h-16 rounded-2xl bg-indigo-600 text-white mb-4 md:mb-6 shadow-xl shadow-indigo-200"
            >
              <Puzzle className="w-6 h-6 md:w-8 md:h-8" />
            </motion.div>
            <motion.h1 
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              className="text-3xl md:text-5xl font-black tracking-tight text-indigo-950 mb-3 md:mb-4"
            >
              CrossWord <span className="text-indigo-600">Pro</span>
            </motion.h1>
            <motion.p 
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.1 }}
              className="text-slate-500 text-base md:text-lg max-w-2xl mx-auto px-4"
            >
              Transform any word list or lengthy text into a professional interlocking crossword puzzle.
            </motion.p>
          </header>

        {/* Input Panel */}
        <div className="bg-white border border-slate-200 rounded-3xl shadow-sm p-6 mb-8 print:hidden">
          <div className="flex p-1 bg-slate-100 rounded-xl mb-6">
            <button 
              onClick={() => setMode('manual')}
              className={`flex-1 py-2.5 px-4 rounded-lg font-bold text-sm transition-all flex items-center justify-center gap-2 ${mode === 'manual' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
            >
              <TypeIcon size={16} /> Manual List
            </button>
            <button 
              onClick={() => setMode('ai')}
              className={`flex-1 py-2.5 px-4 rounded-lg font-bold text-sm transition-all flex items-center justify-center gap-2 ${mode === 'ai' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
            >
              <Sparkles size={16} /> AI Extraction
            </button>
          </div>

          <AnimatePresence mode="wait">
            {mode === 'manual' ? (
              <motion.div 
                key="manual"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
              >
                <label className="block text-sm font-bold text-slate-700 mb-2">Word & Clue Input</label>
                <textarea 
                  value={manualText}
                  onChange={(e) => setManualText(e.target.value)}
                  className="w-full h-48 p-4 bg-slate-50 border border-slate-200 rounded-2xl font-mono text-sm focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 outline-none transition-all"
                  placeholder="WORD : The clue text here..."
                />
                <button 
                  onClick={handleManualGenerate}
                  className="w-full mt-4 py-4 bg-indigo-600 hover:bg-indigo-700 text-white font-black rounded-2xl shadow-lg shadow-indigo-200 transition-all active:scale-[0.98] flex items-center justify-center gap-2"
                >
                  Generate Grid <ChevronRight size={18} />
                </button>
              </motion.div>
            ) : (
              <motion.div 
                key="ai"
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
              >
                <label className="block text-sm font-bold text-slate-700 mb-2">Source Text</label>
                <textarea 
                  value={aiText}
                  onChange={(e) => setAiText(e.target.value)}
                  className="w-full h-48 p-4 bg-slate-50 border border-slate-200 rounded-2xl text-sm focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 outline-none transition-all"
                  placeholder="Paste an article, chapter, or any text here..."
                />
                <div className="flex flex-col sm:flex-row gap-4 mt-4">
                  <div className="flex items-center gap-3 bg-slate-50 border border-slate-200 rounded-2xl px-4 py-2">
                    <Hash size={16} className="text-slate-400" />
                    <span className="text-sm font-bold text-slate-600">Words:</span>
                    <input 
                      type="number" 
                      value={aiCount}
                      onChange={(e) => setAiCount(parseInt(e.target.value) || 5)}
                      className="w-12 bg-transparent font-bold text-indigo-600 outline-none"
                      min={5} max={20}
                    />
                  </div>
                  <button 
                    onClick={handleAiExtract}
                    disabled={loading}
                    className="flex-1 py-4 bg-slate-900 border border-slate-900 hover:bg-indigo-600 hover:border-indigo-600 text-white font-black rounded-2xl shadow-lg transition-all active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {loading ? <Loader2 className="animate-spin" size={18} /> : <Sparkles size={18} />}
                    {loading ? "Analyzing..." : "Analyze & Extract"}
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {error && (
            <motion.div 
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              className="mt-6 p-4 bg-rose-50 border border-rose-200 rounded-2xl text-rose-700 flex items-start gap-3"
            >
              <AlertCircle className="shrink-0 mt-0.5" size={18} />
              <div>
                <p className="text-sm font-bold">Extraction Error</p>
                <p className="text-xs opacity-90">{error}</p>
              </div>
            </motion.div>
          )}
        </div>

        {/* Failed Words Alert */}
        {gridResult && gridResult.failedWords.length > 0 && (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8 p-4 bg-amber-50 border border-amber-200 rounded-2xl text-amber-800 flex items-start gap-3 print:hidden"
          >
            <AlertCircle className="shrink-0 mt-0.5" size={18} />
            <div>
              <p className="text-sm font-bold">Partial Fit Warning</p>
              <p className="text-sm opacity-90">Could not fit {gridResult.failedWords.length} word(s): <span className="font-mono font-bold">{gridResult.failedWords.join(', ')}</span>. Try adding more vocabulary to help interlocking.</p>
            </div>
          </motion.div>
        )}

        {/* Controls */}
        {gridResult && (
          <div className="flex flex-wrap gap-3 mb-8 print:hidden">
            <button 
              onClick={() => setShowKey(!showKey)}
              className={`flex-1 min-w-[140px] py-3 px-6 rounded-2xl font-bold text-sm transition-all flex items-center justify-center gap-2 border-2 ${showKey ? 'bg-emerald-600 border-emerald-600 text-white' : 'bg-white border-slate-200 text-slate-700 hover:border-emerald-500 hover:text-emerald-600'}`}
            >
              <Key size={16} /> {showKey ? "Hide Answers" : "Reveal All"}
            </button>
            <button 
              onClick={handleRevealWord}
              disabled={!activeClue}
              className="flex-1 min-w-[140px] py-3 px-6 bg-white border-2 border-slate-200 hover:border-indigo-500 text-slate-700 hover:text-indigo-600 font-bold text-sm rounded-2xl transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Sparkles size={16} /> Reveal Word
            </button>
            <button 
              onClick={() => setUserInput({})}
              className="flex-1 min-w-[140px] py-3 px-6 bg-white border-2 border-slate-200 hover:border-rose-300 text-slate-700 hover:text-rose-600 font-bold text-sm rounded-2xl transition-all flex items-center justify-center gap-2"
            >
              <RefreshCcw size={16} /> Clear Grid
            </button>
            <button 
              onClick={() => window.print()}
              className="py-3 px-6 bg-white border-2 border-slate-200 hover:border-indigo-500 text-slate-700 hover:text-indigo-600 font-bold text-sm rounded-2xl transition-all flex items-center justify-center gap-2"
            >
              <Printer size={16} />
            </button>
            <button 
              onClick={handleShare}
              className={`py-3 px-6 border-2 font-bold text-sm rounded-2xl transition-all flex items-center justify-center gap-2 ${shareSuccess ? 'bg-indigo-600 border-indigo-600 text-white' : 'bg-white border-slate-200 text-slate-700 hover:border-indigo-500 hover:text-indigo-600'}`}
            >
              {shareSuccess ? <Check size={16} /> : <Share2 size={16} />}
              {shareSuccess ? "Copied!" : "Share Link"}
            </button>
          </div>
        )}

        {/* Output */}
        {gridResult && (
          <motion.div 
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white border border-slate-200 rounded-[2rem] md:rounded-[3rem] p-4 xs:p-6 md:p-12 shadow-xl print:shadow-none print:border-none print:p-0 print-no-overflow"
          >
            <div className="hidden print:block text-center mb-12 border-b-2 border-slate-900 pb-8">
              <h1 className="text-3xl font-black mb-2">CrossWord Master</h1>
              <div className="flex justify-between text-sm font-bold text-slate-600">
                <span>Name: _______________________</span>
                <span>Date: ______________</span>
              </div>
            </div>

            {activeClue && (
              <motion.div 
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-8 p-4 bg-indigo-600 text-white rounded-2xl shadow-lg flex items-center gap-4 print:hidden"
              >
                <div className="shrink-0 w-10 h-10 flex items-center justify-center bg-white/20 rounded-xl font-black text-lg">
                  {activeClue.num}
                </div>
                <div>
                  <p className="text-xs font-bold uppercase tracking-widest opacity-70 mb-0.5">{focusedCell?.dir} CLUE</p>
                  <p className="text-lg font-bold leading-tight">{activeClue.clue}</p>
                </div>
              </motion.div>
            )}

            <div className="w-full overflow-x-auto print:overflow-visible print:mb-8 print-avoid-break pb-8 touch-pan-x">
              <div className="flex min-w-full px-4 sm:px-8">
                <div 
                  className="grid p-1 sm:p-[6px] border-[3px] border-slate-900 bg-slate-900 gap-[2px] sm:gap-[4px] print:bg-black print:border-black print:gap-[2px] [print-color-adjust:exact] min-w-max mx-auto origin-top transition-transform print-scale-fix" 
                  style={{ 
                    gridTemplateColumns: `repeat(${gridResult.cols}, minmax(0, 1fr))`,
                    // Dynamic print scaling: ensure it fits within ~650px print width
                    '--print-scale': gridResult.cols > 15 ? Math.min(1, 650 / (gridResult.cols * 44)) : 1
                  } as any}
                >
                  {gridResult.grid.map((row, y) => row.map((char, x) => {
                    const numberEntry = gridResult.placedWords.find(pw => pw.x === x && pw.y === y);
                    const isFocused = focusedCell?.x === x && focusedCell?.y === y;
                    const isActiveWord = isCellInActiveWord(x, y);
                    const value = userInput[`${x},${y}`] || "";
                    const isCorrect = value === char;
                    
                    return (
                      <div 
                        key={`${x}-${y}`} 
                        onClick={() => handleCellClick(x, y)}
                        className={`relative w-9 h-9 xs:w-11 xs:h-11 sm:w-14 sm:h-14 print:w-10 print:h-10 flex items-center justify-center transition-all cursor-text [print-color-adjust:exact] shadow-[0_2px_0_0_rgba(0,0,0,0.05)]
                          ${char === null ? 'bg-slate-900 print:bg-black' : 
                            isFocused ? 'bg-yellow-200 ring-2 ring-yellow-400 z-20 rounded-lg sm:rounded-xl shadow-md' : 
                            isActiveWord ? 'bg-indigo-50 print:bg-white rounded-lg sm:rounded-xl' : 'bg-white rounded-lg sm:rounded-xl hover:bg-slate-50 hover:scale-[1.02]'}
                          ${char !== null ? 'border-b-2 border-slate-100 print:border-none' : ''}`}
                      >
                        {numberEntry && (
                          <span className="absolute top-1 left-1 sm:top-1.5 sm:left-1.5 text-[8px] xs:text-[9px] sm:text-[12px] print:text-[8px] font-black leading-none text-slate-900 select-none z-10 print:text-black">
                            {numberEntry.num}
                          </span>
                        )}
                        {char !== null && (
                          <input
                            type="text"
                            maxLength={1}
                            data-coord={`${x},${y}`}
                            inputMode="text"
                            autoCapitalize="characters"
                            autoCorrect="off"
                            autoComplete="off"
                            value={showKey ? (char || "") : value}
                            onChange={(e) => handleInputChange(x, y, e.target.value)}
                            onKeyDown={(e) => handleKeyDown(e, x, y)}
                            readOnly={showKey}
                            className={`w-full h-full bg-transparent text-center text-lg xs:text-xl sm:text-3xl print:text-xl font-black focus:outline-none uppercase caret-transparent [print-color-adjust:exact]
                              ${showKey ? 'text-indigo-600 print:text-black' : 
                                value && !isCorrect && !showKey ? 'text-rose-500 print:text-black' : 
                                value && isCorrect ? 'text-indigo-600 print:text-black' : 'text-slate-900 print:text-black'}`}
                          />
                        )}
                      </div>
                    );
                  }))}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-12 border-t-2 border-slate-100 pt-12 print:border-slate-900">
              <section>
                <div className="flex items-center gap-3 mb-6">
                  <div className="p-2 bg-indigo-50 text-indigo-600 rounded-lg">
                    <ListIcon size={18} />
                  </div>
                  <h3 className="text-xl font-black text-slate-900 uppercase tracking-wider">Across</h3>
                </div>
                <ul className="space-y-4">
                  {acrossClues.map((clue, i) => {
                    const isActiveWord = activeClue === clue;
                    return (
                      <li 
                        key={i} 
                        onClick={() => setFocusedCell({ x: clue.x, y: clue.y, dir: 'across' })}
                        className={`flex gap-4 group cursor-pointer p-2 rounded-xl transition-all print-avoid-break ${isActiveWord ? 'bg-indigo-50 ring-1 ring-indigo-100' : 'hover:bg-slate-50'}`}
                      >
                        <span className={`shrink-0 w-8 h-8 flex items-center justify-center rounded-lg font-black text-xs transition-all ${isActiveWord ? 'bg-indigo-600 text-white' : 'bg-slate-100 text-slate-500 group-hover:bg-indigo-600 group-hover:text-white'}`}>
                          {clue.num}
                        </span>
                        <p className={`text-slate-700 font-medium leading-relaxed pt-1 ${isActiveWord ? 'text-indigo-900' : ''}`}>{clue.clue}</p>
                      </li>
                    );
                  })}
                </ul>
              </section>

              <section>
                <div className="flex items-center gap-3 mb-6">
                  <div className="p-2 bg-indigo-50 text-indigo-600 rounded-lg">
                    <ListIcon size={18} />
                  </div>
                  <h3 className="text-xl font-black text-slate-900 uppercase tracking-wider">Down</h3>
                </div>
                <ul className="space-y-4">
                  {downClues.map((clue, i) => {
                    const isActiveWord = activeClue === clue;
                    return (
                      <li 
                        key={i} 
                        onClick={() => setFocusedCell({ x: clue.x, y: clue.y, dir: 'down' })}
                        className={`flex gap-4 group cursor-pointer p-2 rounded-xl transition-all print-avoid-break ${isActiveWord ? 'bg-indigo-50 ring-1 ring-indigo-100' : 'hover:bg-slate-50'}`}
                      >
                        <span className={`shrink-0 w-8 h-8 flex items-center justify-center rounded-lg font-black text-xs transition-all ${isActiveWord ? 'bg-indigo-600 text-white' : 'bg-slate-100 text-slate-500 group-hover:bg-indigo-600 group-hover:text-white'}`}>
                          {clue.num}
                        </span>
                        <p className={`text-slate-700 font-medium leading-relaxed pt-1 ${isActiveWord ? 'text-indigo-900' : ''}`}>{clue.clue}</p>
                      </li>
                    );
                  })}
                </ul>
              </section>
            </div>
          </motion.div>
        )}

      </div>
      
      {/* Footer Branding */}
      <footer className="mt-20 text-center pb-12 print:hidden">
        <p className="text-sm font-bold text-slate-400 uppercase tracking-widest flex items-center justify-center gap-2">
          Powered by <Sparkles size={14} /> Gemini Intelligence
        </p>
      </footer>
    </div>
  );
}

