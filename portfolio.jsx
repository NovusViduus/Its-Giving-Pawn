import { useState, useEffect, useRef } from "react";

// ═══ THEME ═══
const ACCENT = "#00ff88";
const ACCENT2 = "#ff6b35";
const ACCENT3 = "#a78bfa";
const BG = "#0a0a0f";
const CARD = "#12121a";
const TEXT = "#e8e8ed";
const MUTED = "#8888a0";

// ═══ CHESS API CONFIG ═══
// Update this URL after deploying to Railway
const CHESS_API = "https://its-giving-pawn-production.up.railway.app";

// ═══ DATA ═══
const projects = [
  { title:"It's Giving Pawn", icon:"♟️", tag:"Chess Engine", tech:["Python","NumPy","chess"], color:ACCENT,
    short:"UCI-compliant chess engine with 12-component evaluation and advanced search algorithms.",
    long:"Built a full UCI-compliant chess engine featuring Principal Variation Search with alpha-beta pruning, transposition tables using Zobrist hashing, Late Move Reductions, null move pruning, and quiescence search. Developed a 12-feature position evaluator with phase-aware weighting for king safety, pawn structure, piece mobility, and passed pawn detection. Integrated Polyglot opening book with pattern detection for 8 common openings. The engine never lost a single game against random or mate-in-one bots across 400 test games.",
    github:"https://github.com/NovusViduus/Its-Giving-Pawn" },
  { title:"Liquidity Mainframe", icon:"📊", tag:"BI Dashboard", tech:["Python","Gradio","Plotly","yfinance"], color:ACCENT2,
    short:"Real-time financial analytics dashboard with cyberpunk theming and automated signal detection.",
    long:"Built a real-time financial analytics dashboard integrating yfinance and Finnhub APIs with 10+ interactive Plotly chart types and custom cyberpunk theming. Implemented technical indicators including RSI, MACD, Bollinger Bands, and moving averages with automated bullish/bearish signal detection and priority-ranked insights. Designed multi-source data pipeline supporting CSV/Excel uploads and live API fetching with 15-minute caching layer for rate limit compliance." },
  { title:"Image Mosaic Generator", icon:"🖼️", tag:"Computer Vision", tech:["Python","NumPy","PIL","Gradio"], color:ACCENT3,
    short:"High-performance photo mosaic generator with 10x speedup over naive implementation.",
    long:"Developed a photo mosaic generator using vectorized NumPy with memory-optimized batch processing in 10K cell chunks, achieving 10x+ speedup over naive implementation. Built a tile caching system storing pre-computed RGB averages at multiple grid sizes, eliminating redundant image processing across generations." },
];

const journey = [
  { role:"Teaching Assistant — CS 5130", org:"Northeastern University", period:"Current", type:"edu" },
  { role:"M.S. Artificial Intelligence", org:"Northeastern University, Seattle", period:"2025 – Dec 2026", type:"edu" },
  { role:"Certificate, CS Fundamentals", org:"Seattle University", period:"2025", type:"edu" },
  { role:"Institutional Giving Manager", org:"Tacoma Art Museum", period:"2024", type:"work", detail:"Migrated grant systems to CRM, managed $1.5M pipeline across 60+ applications" },
  { role:"Grants Associate", org:"Committee For Children", period:"2022–2023", type:"work", detail:"Built CRM with 300+ orgs, cataloged 500+ funding opportunities" },
  { role:"M.S. Global Health Policy", org:"Brandeis University", period:"2021", type:"edu" },
  { role:"B.S. Anthropology & Public Health", org:"Southern Illinois University", period:"2018", type:"edu" },
];

const skills = ["Python","C++","Java","SQL","NumPy","Pandas","Plotly","Git","Linux","Data Structures","Algorithms","Machine Learning"];
const NAV = ["About","Projects","Journey","Contact"];

// ═══ CHESS GAME (with API integration) ═══
const pieceMap = {k:"♚",K:"♔",q:"♛",Q:"♕",r:"♜",R:"♖",b:"♝",B:"♗",n:"♞",N:"♘",p:"♟",P:"♙"};

function fenToBoard(fen) {
  const rows = fen.split(" ")[0].split("/");
  return rows.map(row => {
    const cells = [];
    for (const ch of row) {
      if ("12345678".includes(ch)) for (let i=0;i<parseInt(ch);i++) cells.push("");
      else cells.push(ch);
    }
    return cells;
  });
}

function boardToFen(board, turn, castling, ep, halfmove, fullmove) {
  const rows = board.map(row => {
    let s="", empty=0;
    for (const c of row) { if(c===""){empty++} else {if(empty>0){s+=empty;empty=0}s+=c} }
    if(empty>0) s+=empty;
    return s;
  });
  return `${rows.join("/")} ${turn} ${castling} ${ep} ${halfmove} ${fullmove}`;
}

function MiniChess({ onClose }) {
  const [fen, setFen] = useState("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  const [board, setBoard] = useState(fenToBoard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));
  const [sel, setSel] = useState(null);
  const [turn, setTurn] = useState("w");
  const [msg, setMsg] = useState("Your move (white). Click a piece.");
  const [thinking, setThinking] = useState(false);
  const [useAPI, setUseAPI] = useState(false);
  const [apiStatus, setApiStatus] = useState("unchecked");

  const isW = p => p && p === p.toUpperCase();
  const own = p => turn==="w" ? isW(p) : (p && !isW(p));

  // Check if API is available
  useEffect(() => {
    fetch(`${CHESS_API}/health`).then(r => r.json())
      .then(d => { if(d.status==="alive") { setApiStatus("online"); setUseAPI(true); } })
      .catch(() => setApiStatus("offline"));
  }, []);

  const getAIMove = async (currentFen) => {
    if (!useAPI || apiStatus !== "online") {
      // Fallback: random move
      return null;
    }
    try {
      const res = await fetch(`${CHESS_API}/move`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen: currentFen, depth: 3, time_limit: 2.0 }),
      });
      const data = await res.json();
      return data;
    } catch {
      return null;
    }
  };

  const makeRandomBlackMove = (currentBoard) => {
    const blackPieces = [];
    currentBoard.forEach((row,r) => row.forEach((p,c) => {
      if (p && p===p.toLowerCase()) blackPieces.push([r,c]);
    }));
    if (blackPieces.length === 0) return;
    const shuffled = blackPieces.sort(() => Math.random()-0.5);
    for (const [r,c] of shuffled) {
      const targets = [];
      for (let tr=0;tr<8;tr++) for(let tc=0;tc<8;tc++) {
        if (currentBoard[tr][tc]==="" || isW(currentBoard[tr][tc])) targets.push([tr,tc]);
      }
      if (targets.length > 0) {
        const [tr,tc] = targets[Math.floor(Math.random()*targets.length)];
        const nb = currentBoard.map(x=>[...x]);
        nb[tr][tc] = nb[r][c]; nb[r][c] = "";
        setBoard(nb); setTurn("w"); setMsg("Your move (white).");
        return;
      }
    }
  };

  const click = async (r,c) => {
    if (turn !== "w" || thinking) return;
    if (sel) {
      const [sr,sc] = sel;
      if (sr===r && sc===c) { setSel(null); return; }
      if (own(board[r][c])) { setSel([r,c]); return; }

      // Make player's move
      const nb = board.map(x=>[...x]);
      const cap = nb[r][c];
      nb[r][c] = nb[sr][sc]; nb[sr][sc] = "";
      setBoard(nb); setSel(null);

      if (cap==="k") { setMsg("Checkmate! You win!"); setTurn("done"); return; }

      setTurn("b");
      setMsg("Engine thinking...");
      setThinking(true);

      // Try API move
      if (useAPI && apiStatus === "online") {
        // Build rough FEN from board state
        const newFen = boardToFen(nb, "b", "KQkq", "-", "0", "1");
        const aiResponse = await getAIMove(newFen);
        if (aiResponse && aiResponse.move) {
          // Parse UCI move
          const from = [8 - parseInt(aiResponse.move[1]), aiResponse.move.charCodeAt(0) - 97];
          const to = [8 - parseInt(aiResponse.move[3]), aiResponse.move.charCodeAt(2) - 97];
          const nb2 = nb.map(x=>[...x]);
          const cap2 = nb2[to[0]][to[1]];
          nb2[to[0]][to[1]] = nb2[from[0]][from[1]]; nb2[from[0]][from[1]] = "";
          setBoard(nb2);
          if (cap2==="K") { setMsg("Engine wins! The necromancer falls."); setTurn("done"); }
          else { setMsg(`Engine played ${aiResponse.move} (${aiResponse.thinking_time}s). Your move.`); setTurn("w"); }
          setThinking(false);
          return;
        }
      }

      // Fallback to random
      setTimeout(() => {
        makeRandomBlackMove(nb);
        setThinking(false);
      }, 500);
    } else if (board[r][c] && own(board[r][c])) {
      setSel([r,c]);
    }
  };

  return (
    <div style={{ position:"fixed", inset:0, zIndex:10000, background:"rgba(0,0,0,0.9)", backdropFilter:"blur(12px)", display:"flex", alignItems:"center", justifyContent:"center", flexDirection:"column" }}>
      <div style={{ background:"#0c0c1a", border:`1px solid ${ACCENT}33`, borderRadius:16, padding:28, position:"relative", boxShadow:`0 0 80px ${ACCENT}12` }}>
        <button onClick={onClose} style={{ position:"absolute", top:12, right:16, background:"none", border:"none", color:MUTED, fontSize:20, cursor:"pointer" }}>✕</button>
        <h3 style={{ fontFamily:"'Space Grotesk',sans-serif", fontSize:18, fontWeight:700, color:ACCENT, margin:"0 0 4px", textAlign:"center" }}>♟️ It's Giving Pawn</h3>
        <p style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:11, color:MUTED, textAlign:"center", margin:"0 0 4px" }}>{msg}</p>
        <p style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:10, textAlign:"center", margin:"0 0 14px",
          color: apiStatus==="online" ? ACCENT : apiStatus==="offline" ? ACCENT2 : MUTED }}>
          {apiStatus==="online" ? "⚡ live engine connected" : apiStatus==="offline" ? "⚠ engine offline — random moves" : "checking engine..."}
        </p>
        <div style={{ display:"grid", gridTemplateColumns:"repeat(8,44px)", gridTemplateRows:"repeat(8,44px)", border:`2px solid ${ACCENT}33`, borderRadius:4, overflow:"hidden" }}>
          {board.map((row,r) => row.map((pc,c) => {
            const dk = (r+c)%2===1, isSel = sel&&sel[0]===r&&sel[1]===c;
            return <div key={`${r}${c}`} onClick={()=>click(r,c)} style={{
              width:44, height:44, display:"flex", alignItems:"center", justifyContent:"center", fontSize:26, cursor: turn==="w"?"pointer":"default",
              background: isSel ? ACCENT+"44" : dk ? "#101025" : "#181838", transition:"background 0.15s",
              boxShadow: isSel ? `inset 0 0 14px ${ACCENT}55` : "none",
            }}
            onMouseEnter={e=>{ if(!isSel&&turn==="w") e.target.style.background=ACCENT+"15"; }}
            onMouseLeave={e=>{ if(!isSel) e.target.style.background=dk?"#101025":"#181838"; }}
            >{pc && <span style={{ filter: isW(pc) ? "drop-shadow(0 0 4px #ffffff77)" : `drop-shadow(0 0 4px ${ACCENT3}77)` }}>{pieceMap[pc]}</span>}</div>;
          }))}
        </div>
        {thinking && <div style={{ textAlign:"center", marginTop:12 }}>
          <div style={{ display:"inline-block", width:16, height:16, border:`2px solid ${ACCENT}`, borderTopColor:"transparent", borderRadius:"50%", animation:"spin 0.8s linear infinite" }}/>
        </div>}
        <p style={{ fontSize:10, color:MUTED+"77", textAlign:"center", marginTop:12, fontFamily:"'JetBrains Mono',monospace" }}>
          {useAPI ? "powered by 12-component evaluation engine" : "no move validation — play with honor"}
        </p>
      </div>
    </div>
  );
}

// ═══ COMPONENTS ═══
function Cursor() {
  const [pos, setPos] = useState({x:-100,y:-100});
  const [click, setClick] = useState(false);
  const trail = useRef([]);
  const [tp, setTp] = useState([]);
  useEffect(()=>{
    const m=e=>{ setPos({x:e.clientX,y:e.clientY}); trail.current=[{x:e.clientX,y:e.clientY},...trail.current].slice(0,5); setTp([...trail.current]); };
    const d=()=>setClick(true); const u=()=>setClick(false);
    window.addEventListener("mousemove",m); window.addEventListener("mousedown",d); window.addEventListener("mouseup",u);
    return()=>{ window.removeEventListener("mousemove",m); window.removeEventListener("mousedown",d); window.removeEventListener("mouseup",u); };
  },[]);
  return <>
    <style>{`@media (pointer: fine) { * { cursor: none !important; } }`}</style>
    {tp.map((p,i)=><div key={i} style={{ position:"fixed", left:p.x-2, top:p.y-2, width:4-i*0.5, height:4-i*0.5, borderRadius:"50%", background:ACCENT, opacity:0.2-i*0.035, pointerEvents:"none", zIndex:99998 }}/>)}
    <div style={{ position:"fixed", left:pos.x-16, top:pos.y-16, width:32, height:32, borderRadius:"50%", border:`1.5px solid ${ACCENT}55`, pointerEvents:"none", zIndex:99999, transition:"transform 0.1s", transform:click?"scale(0.6)":"scale(1)" }}/>
    <div style={{ position:"fixed", left:pos.x-3, top:pos.y-3, width:6, height:6, borderRadius:"50%", background:ACCENT, boxShadow:`0 0 10px ${ACCENT}88`, pointerEvents:"none", zIndex:99999, transition:"transform 0.05s", transform:click?"scale(1.5)":"scale(1)" }}/>
  </>;
}

function TypeWriter({ phrases }) {
  const [ci, setCi] = useState(0);
  const [txt, setTxt] = useState("");
  const [del, setDel] = useState(false);
  useEffect(()=>{
    const p=phrases[ci], spd=del?35:70;
    const to=setTimeout(()=>{
      if(!del){ setTxt(p.slice(0,txt.length+1)); if(txt.length===p.length) setTimeout(()=>setDel(true),2200); }
      else { setTxt(p.slice(0,txt.length-1)); if(txt.length===0){ setDel(false); setCi((ci+1)%phrases.length); } }
    },spd);
    return()=>clearTimeout(to);
  },[txt,del,ci,phrases]);
  return <span>{txt}<span style={{ display:"inline-block", width:2, height:"1em", background:ACCENT, marginLeft:2, animation:"blink 1s step-end infinite", verticalAlign:"text-bottom" }}/></span>;
}

function Reveal({ children, delay=0 }) {
  const ref=useRef(null); const [vis,setVis]=useState(false);
  useEffect(()=>{ const el=ref.current; if(!el) return; const o=new IntersectionObserver(([e])=>{ if(e.isIntersecting)setVis(true); },{threshold:0.1}); o.observe(el); return()=>o.disconnect(); },[]);
  return <div ref={ref} style={{ opacity:vis?1:0, transform:vis?"none":"translateY(40px)", transition:`opacity 0.7s ease ${delay}s, transform 0.7s ease ${delay}s` }}>{children}</div>;
}

function ProjectCard({ p }) {
  const [exp, setExp] = useState(false);
  const [hov, setHov] = useState(false);
  return (
    <div onClick={()=>setExp(!exp)} onMouseEnter={()=>setHov(true)} onMouseLeave={()=>setHov(false)}
      style={{ background:hov?"#161622":CARD, border:`1px solid ${hov||exp?p.color+"44":"#ffffff0d"}`, borderRadius:16, padding:"28px 24px", transition:"all 0.4s cubic-bezier(0.16,1,0.3,1)", transform:hov?"translateY(-4px)":"none", cursor:"pointer", position:"relative", overflow:"hidden", boxShadow:hov?`0 8px 40px ${p.color}11`:"none" }}>
      <div style={{ position:"absolute", top:0, left:0, right:0, height:2, background:`linear-gradient(90deg,transparent,${p.color},transparent)`, opacity:hov||exp?0.8:0, transition:"opacity 0.3s" }}/>
      <div style={{ display:"flex", alignItems:"center", gap:12, marginBottom:12 }}>
        <span style={{ fontSize:28 }}>{p.icon}</span>
        <span style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:10, color:p.color, background:p.color+"15", padding:"4px 12px", borderRadius:100, letterSpacing:1.5, textTransform:"uppercase" }}>{p.tag}</span>
      </div>
      <h3 style={{ fontFamily:"'Space Grotesk','Syne',sans-serif", fontSize:22, fontWeight:700, color:TEXT, margin:"0 0 8px" }}>{p.title}</h3>
      <div style={{ display:"flex", gap:6, flexWrap:"wrap", marginBottom:12 }}>
        {p.tech.map(t=><span key={t} style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:10, color:MUTED, padding:"3px 8px", border:"1px solid #ffffff0d", borderRadius:6 }}>{t}</span>)}
      </div>
      <p style={{ fontSize:14, lineHeight:1.7, color:"#b0b0c0", margin:0, fontFamily:"'DM Sans',sans-serif" }}>{exp?p.long:p.short}</p>
      <div style={{ marginTop:12, display:"flex", justifyContent:"space-between", alignItems:"center" }}>
        <span style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:11, color:p.color, opacity:0.6, display:"flex", alignItems:"center", gap:6 }}>
          <span style={{ transform:exp?"rotate(90deg)":"none", transition:"transform 0.3s", display:"inline-block" }}>▸</span>
          {exp?"collapse":"expand details"}
        </span>
        {p.github && <a href={p.github} target="_blank" rel="noopener noreferrer" onClick={e=>e.stopPropagation()}
          style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:11, color:MUTED, textDecoration:"none", transition:"color 0.2s" }}
          onMouseEnter={e=>e.target.style.color=ACCENT} onMouseLeave={e=>e.target.style.color=MUTED}>
          github →
        </a>}
      </div>
    </div>
  );
}

function OriginModal({ onClose }) {
  const paras = [
    "I didn't start in tech. I started in a completely different world — anthropology, public health, global health policy. I studied the way systems fail people and the way communities find resilience in the margins.",
    "For years I worked in grants and institutional development. I built databases, tracked pipelines, managed millions in funding, and coordinated across departments. I was good at it. But I kept bumping into the same wall: the tools I was using weren't built for the problems I was seeing.",
    "So I started building my own.",
    "It began with a curiosity — a CS fundamentals certificate at Seattle University. Then it became a conviction — a Master's in AI at Northeastern. The transition wasn't graceful. I went from being an expert in my field to being a beginner again. I solved LeetCode problems at 2 AM. I built a chess engine to prove to myself I could think like a programmer.",
    "Now I write code that solves problems I spent years studying from the other side. The anthropologist in me sees the human system. The engineer in me builds the solution.",
    "I'm not a traditional candidate. I never will be. That's the point.",
  ];
  return (
    <div style={{ position:"fixed", inset:0, zIndex:9999, background:BG+"f5", backdropFilter:"blur(20px)", display:"flex", alignItems:"center", justifyContent:"center", padding:40, overflowY:"auto" }}>
      <div style={{ maxWidth:640, width:"100%" }}>
        <button onClick={onClose} style={{ background:"none", border:"1px solid #ffffff0d", color:MUTED, padding:"8px 20px", borderRadius:100, fontFamily:"'JetBrains Mono',monospace", fontSize:12, cursor:"pointer", marginBottom:40, transition:"all 0.2s" }}
          onMouseEnter={e=>{e.target.style.borderColor=ACCENT;e.target.style.color=ACCENT;}} onMouseLeave={e=>{e.target.style.borderColor="#ffffff0d";e.target.style.color=MUTED;}}>← back</button>
        <h2 style={{ fontFamily:"'Syne','Space Grotesk',sans-serif", fontSize:36, fontWeight:800, color:ACCENT, margin:"0 0 12px", letterSpacing:-1 }}>The Origin</h2>
        <div style={{ width:60, height:2, margin:"0 0 40px", background:`linear-gradient(90deg,${ACCENT},${ACCENT2})` }}/>
        {paras.map((p,i)=><p key={i} style={{
          fontSize:i===2?20:17, fontWeight:i===2?600:400, lineHeight:1.9,
          color:i===2?ACCENT:i===5?ACCENT2:TEXT,
          fontFamily:i===2?"'Syne',sans-serif":"'DM Sans',sans-serif",
          margin:"0 0 28px", fontStyle:i===5?"italic":"normal",
          opacity:0, animation:`fadeUp 0.6s ease ${0.2+i*0.15}s forwards`,
        }}>{p}</p>)}
      </div>
    </div>
  );
}

// ═══ PARTICLES ═══
function Particles() {
  return <div style={{ position:"fixed", inset:0, pointerEvents:"none", zIndex:0 }}>
    {Array.from({length:25}).map((_,i)=>{
      const size=Math.random()*2.5+1, left=Math.random()*100, dur=Math.random()*20+15, y=Math.random()*100;
      return <div key={i} style={{
        position:"absolute", width:size, height:size, borderRadius:"50%",
        background:Math.random()>0.5?ACCENT:ACCENT2,
        opacity:Math.random()*0.2+0.05, left:`${left}%`, top:`${y}%`,
        animation:`float ${dur}s ease-in-out ${i*0.5}s infinite alternate`, pointerEvents:"none",
      }}/>;
    })}
  </div>;
}

// ═══ MAIN ═══
export default function Portfolio() {
  const [loaded, setLoaded] = useState(false);
  const [chess, setChess] = useState(false);
  const [reading, setReading] = useState(false);
  const [active, setActive] = useState("about");
  const [mousePos, setMousePos] = useState({x:0,y:0});

  useEffect(()=>{
    setTimeout(()=>setLoaded(true),100);
    const l=document.createElement("link");
    l.href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap";
    l.rel="stylesheet"; document.head.appendChild(l);
  },[]);

  useEffect(()=>{
    const obs=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting)setActive(e.target.id);});},{threshold:0.3});
    NAV.forEach(n=>{const el=document.getElementById(n.toLowerCase());if(el)obs.observe(el);});
    return()=>obs.disconnect();
  },[]);

  const scrollTo=id=>document.getElementById(id.toLowerCase())?.scrollIntoView({behavior:"smooth"});

  return (
    <div onMouseMove={e=>setMousePos({x:e.clientX,y:e.clientY})} style={{ minHeight:"100vh", background:BG, color:TEXT, fontFamily:"'DM Sans',sans-serif", overflowX:"hidden", position:"relative" }}>
      <style>{`
        @keyframes blink{0%,50%{opacity:1}51%,100%{opacity:0}}
        @keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
        @keyframes pulse{0%,100%{opacity:0.4}50%{opacity:1}}
        @keyframes float{0%{transform:translateY(0) translateX(0)}100%{transform:translateY(-25px) translateX(12px)}}
        @keyframes grad{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
        @keyframes spin{to{transform:rotate(360deg)}}
        ::selection{background:${ACCENT}44}
        *{box-sizing:border-box}
        html{scroll-behavior:smooth}
      `}</style>

      <Particles/>
      <Cursor/>
      {chess && <MiniChess onClose={()=>setChess(false)}/>}
      {reading && <OriginModal onClose={()=>setReading(false)}/>}

      {/* Cursor glow */}
      <div style={{ position:"fixed", width:350, height:350, borderRadius:"50%", background:`radial-gradient(circle,${ACCENT}06 0%,transparent 70%)`, transform:`translate(${mousePos.x-175}px,${mousePos.y-175}px)`, pointerEvents:"none", zIndex:0, transition:"transform 0.15s ease-out" }}/>

      {/* NAV */}
      <nav style={{ position:"fixed", top:0, left:0, right:0, zIndex:1000, padding:"14px 36px", display:"flex", justifyContent:"space-between", alignItems:"center", backdropFilter:"blur(20px)", background:BG+"cc", borderBottom:"1px solid #ffffff08" }}>
        <div style={{ fontFamily:"'Syne',sans-serif", fontWeight:800, fontSize:20, background:`linear-gradient(135deg,${ACCENT},${ACCENT2})`, backgroundSize:"200% 200%", animation:"grad 4s ease infinite", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>GH</div>
        <div style={{ display:"flex", gap:22, alignItems:"center" }}>
          {NAV.map(n=><button key={n} onClick={()=>scrollTo(n)} style={{ background:"none", border:"none", fontFamily:"'JetBrains Mono',monospace", fontSize:11, color:active===n.toLowerCase()?ACCENT:MUTED, letterSpacing:1.5, textTransform:"uppercase", cursor:"pointer", transition:"color 0.3s", padding:"4px 0", borderBottom:active===n.toLowerCase()?`1px solid ${ACCENT}`:"1px solid transparent" }}
            onMouseEnter={e=>e.target.style.color=ACCENT} onMouseLeave={e=>{if(active!==n.toLowerCase())e.target.style.color=MUTED;}}>{n}</button>)}
          <button onClick={()=>setChess(true)} style={{ background:"none", border:"1px solid #ffffff0d", borderRadius:8, padding:"6px 10px", fontSize:16, cursor:"pointer", transition:"all 0.3s" }} title="Play my chess engine"
            onMouseEnter={e=>{e.target.style.borderColor=ACCENT;e.target.style.boxShadow=`0 0 12px ${ACCENT}33`;}} onMouseLeave={e=>{e.target.style.borderColor="#ffffff0d";e.target.style.boxShadow="none";}}>♟️</button>
        </div>
      </nav>

      <div style={{ maxWidth:900, margin:"0 auto", padding:"0 32px", position:"relative", zIndex:1 }}>

        {/* HERO */}
        <section id="about" style={{ minHeight:"100vh", display:"flex", flexDirection:"column", justifyContent:"center", paddingTop:80 }}>
          <div style={{ opacity:loaded?1:0, transform:loaded?"none":"translateY(40px)", transition:"all 1s ease 0.2s" }}>
            <div style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:13, color:ACCENT, marginBottom:20, display:"flex", alignItems:"center", gap:8 }}>
              <span style={{ width:8, height:8, borderRadius:"50%", background:ACCENT, display:"inline-block", animation:"pulse 2s infinite" }}/>
              available for opportunities
            </div>
            <h1 style={{ fontFamily:"'Syne','Space Grotesk',sans-serif", fontSize:"clamp(48px,8vw,80px)", fontWeight:800, lineHeight:1.05, margin:"0 0 24px", letterSpacing:-2 }}>
              Graeme<br/>
              <span style={{ background:`linear-gradient(135deg,${ACCENT},${ACCENT2})`, backgroundSize:"200% 200%", animation:"grad 4s ease infinite", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>Huntley</span>
            </h1>
            <p style={{ fontSize:18, color:MUTED, margin:"0 0 6px", lineHeight:1.6 }}>
              <TypeWriter phrases={["building intelligent systems","chess engine architect","anthropologist turned developer","MS in AI @ Northeastern","turning data into decisions"]}/>
            </p>
            <p style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:14, color:"#666680", margin:"16px 0 40px" }}>Seattle, WA · Python · C++ · Java · ML</p>
            <div style={{ display:"flex", gap:16, flexWrap:"wrap" }}>
              <a href="https://www.linkedin.com/in/graeme-huntley/" target="_blank" rel="noopener noreferrer" style={{ display:"inline-flex", alignItems:"center", gap:8, padding:"12px 28px", borderRadius:100, background:`linear-gradient(135deg,${ACCENT},${ACCENT}cc)`, color:BG, fontWeight:600, fontSize:14, textDecoration:"none", transition:"all 0.2s", boxShadow:`0 4px 20px ${ACCENT}33` }}
                onMouseEnter={e=>{e.target.style.transform="translateY(-2px)";e.target.style.boxShadow=`0 8px 30px ${ACCENT}44`;}} onMouseLeave={e=>{e.target.style.transform="none";e.target.style.boxShadow=`0 4px 20px ${ACCENT}33`;}}>LinkedIn →</a>
              <a href="mailto:huntley.g@northeastern.edu" style={{ display:"inline-flex", alignItems:"center", padding:"12px 28px", borderRadius:100, border:"1px solid #ffffff20", color:TEXT, fontWeight:500, fontSize:14, textDecoration:"none", transition:"all 0.2s" }}
                onMouseEnter={e=>{e.target.style.borderColor=ACCENT+"66";e.target.style.color=ACCENT;}} onMouseLeave={e=>{e.target.style.borderColor="#ffffff20";e.target.style.color=TEXT;}}>Get in touch</a>
            </div>
          </div>
        </section>

        {/* ORIGIN */}
        <Reveal>
          <div style={{ margin:"0 0 100px", padding:"36px 32px", borderRadius:16, position:"relative", overflow:"hidden", background:`linear-gradient(135deg,${ACCENT}08,${ACCENT2}08)`, border:"1px solid #ffffff0a" }}>
            <div style={{ position:"absolute", top:0, left:0, bottom:0, width:3, background:`linear-gradient(to bottom,${ACCENT},${ACCENT2})` }}/>
            <p style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:12, color:ACCENT, margin:"0 0 12px", textTransform:"uppercase", letterSpacing:2 }}>the origin story</p>
            <p style={{ fontSize:16, lineHeight:1.9, color:"#b0b0c0", margin:"0 0 20px" }}>
              I started in anthropology and public health — studying systems, patterns, and the way people make decisions. Then I discovered that code lets you <em style={{ color:ACCENT, fontStyle:"normal" }}>build</em> solutions, not just study them.
            </p>
            <button onClick={()=>setReading(true)} style={{ background:"none", border:`1px solid ${ACCENT}33`, color:ACCENT, padding:"10px 24px", borderRadius:100, fontFamily:"'JetBrains Mono',monospace", fontSize:12, cursor:"pointer", transition:"all 0.3s", letterSpacing:1 }}
              onMouseEnter={e=>{e.target.style.background=ACCENT+"15";e.target.style.borderColor=ACCENT;}} onMouseLeave={e=>{e.target.style.background="none";e.target.style.borderColor=ACCENT+"33";}}>▸ read the full story</button>
          </div>
        </Reveal>

        {/* SKILLS */}
        <Reveal>
          <div style={{ marginBottom:100 }}>
            <h2 style={{ fontFamily:"'Syne',sans-serif", fontSize:14, fontWeight:700, color:ACCENT, textTransform:"uppercase", letterSpacing:3, margin:"0 0 24px" }}>Tech Stack</h2>
            <div style={{ display:"flex", flexWrap:"wrap", gap:4 }}>
              {skills.map(s=><span key={s} style={{ display:"inline-block", fontFamily:"'JetBrains Mono',monospace", fontSize:13, padding:"8px 18px", borderRadius:100, border:"1px solid #ffffff15", color:MUTED, transition:"all 0.3s", cursor:"default", margin:"2px" }}
                onMouseEnter={e=>{e.target.style.borderColor=ACCENT+"88";e.target.style.color=ACCENT;e.target.style.background=ACCENT+"0d";}}
                onMouseLeave={e=>{e.target.style.borderColor="#ffffff15";e.target.style.color=MUTED;e.target.style.background="transparent";}}>{s}</span>)}
            </div>
          </div>
        </Reveal>

        {/* PROJECTS */}
        <section id="projects">
          <Reveal><h2 style={{ fontFamily:"'Syne',sans-serif", fontSize:14, fontWeight:700, color:ACCENT, textTransform:"uppercase", letterSpacing:3, margin:"0 0 40px" }}>Projects</h2></Reveal>
          <div style={{ display:"grid", gap:20, marginBottom:100 }}>
            {projects.map((p,i)=><Reveal key={p.title} delay={i*0.1}><ProjectCard p={p}/></Reveal>)}
          </div>
        </section>

        {/* JOURNEY */}
        <section id="journey">
          <Reveal><h2 style={{ fontFamily:"'Syne',sans-serif", fontSize:14, fontWeight:700, color:ACCENT, textTransform:"uppercase", letterSpacing:3, margin:"0 0 36px" }}>Journey</h2></Reveal>
          <div style={{ marginBottom:100 }}>
            {journey.map((item,i)=><Reveal key={i} delay={i*0.06}>
              <div style={{ display:"flex", gap:20, padding:"18px 0", borderLeft:"2px solid #ffffff12", paddingLeft:24, marginLeft:8, position:"relative", transition:"border-color 0.3s" }}
                onMouseEnter={e=>e.currentTarget.style.borderLeftColor=ACCENT} onMouseLeave={e=>e.currentTarget.style.borderLeftColor="#ffffff12"}>
                <div style={{ position:"absolute", left:-6, top:24, width:10, height:10, borderRadius:"50%", background:item.type==="edu"?ACCENT3:ACCENT, border:`2px solid ${BG}` }}/>
                <div style={{ flex:1 }}>
                  <div style={{ display:"flex", justifyContent:"space-between", flexWrap:"wrap", gap:8, alignItems:"baseline" }}>
                    <h4 style={{ fontFamily:"'DM Sans',sans-serif", fontSize:16, fontWeight:600, color:TEXT, margin:0 }}>{item.role}</h4>
                    <span style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:12, color:item.type==="edu"?ACCENT3:ACCENT, background:(item.type==="edu"?ACCENT3:ACCENT)+"12", padding:"3px 10px", borderRadius:100 }}>{item.period}</span>
                  </div>
                  <p style={{ fontSize:14, color:ACCENT2, margin:"4px 0 0", fontWeight:500, fontFamily:"'DM Sans',sans-serif" }}>{item.org}</p>
                  {item.detail && <p style={{ fontSize:13, color:MUTED, margin:"6px 0 0", lineHeight:1.6 }}>{item.detail}</p>}
                </div>
              </div>
            </Reveal>)}
          </div>
        </section>

        {/* CONTACT */}
        <section id="contact">
          <Reveal>
            <div style={{ textAlign:"center", padding:"80px 0 120px", borderTop:"1px solid #ffffff08" }}>
              <h2 style={{ fontFamily:"'Syne',sans-serif", fontSize:"clamp(32px,5vw,48px)", fontWeight:800, margin:"0 0 16px", letterSpacing:-1 }}>
                Let's{" "}<span style={{ background:`linear-gradient(135deg,${ACCENT},${ACCENT2})`, WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>connect</span>
              </h2>
              <p style={{ color:MUTED, fontSize:16, margin:"0 auto 36px", maxWidth:420, lineHeight:1.7 }}>Open to internships, collaborations, and conversations about AI, chess engines, or anything in between.</p>
              <div style={{ display:"flex", justifyContent:"center", gap:16, flexWrap:"wrap" }}>
                <a href="mailto:huntley.g@northeastern.edu" style={{ padding:"14px 36px", borderRadius:100, background:`linear-gradient(135deg,${ACCENT},${ACCENT}cc)`, color:BG, fontWeight:600, fontSize:15, textDecoration:"none", transition:"all 0.2s" }}
                  onMouseEnter={e=>{e.target.style.transform="translateY(-2px)";e.target.style.boxShadow=`0 8px 30px ${ACCENT}33`;}} onMouseLeave={e=>{e.target.style.transform="none";e.target.style.boxShadow="none";}}>huntley.g@northeastern.edu</a>
                <a href="https://www.linkedin.com/in/graeme-huntley/" target="_blank" rel="noopener noreferrer" style={{ padding:"14px 36px", borderRadius:100, border:"1px solid #ffffff20", color:TEXT, fontWeight:500, fontSize:15, textDecoration:"none", transition:"all 0.2s" }}
                  onMouseEnter={e=>{e.target.style.borderColor=ACCENT+"66";e.target.style.color=ACCENT;}} onMouseLeave={e=>{e.target.style.borderColor="#ffffff20";e.target.style.color=TEXT;}}>LinkedIn</a>
              </div>
              <div style={{ marginTop:80, fontFamily:"'JetBrains Mono',monospace", fontSize:12, color:"#444460" }}>
                <span style={{color:ACCENT+"66"}}>graeme@portfolio</span>
                <span style={{color:"#555"}}>:</span>
                <span style={{color:"#6666aa"}}>~</span>
                <span style={{color:"#555"}}>$ </span>
                <span style={{color:MUTED}}>building the future</span>
                <span style={{ display:"inline-block", width:8, height:16, background:ACCENT, marginLeft:4, animation:"blink 1s step-end infinite", verticalAlign:"text-bottom" }}/>
              </div>
            </div>
          </Reveal>
        </section>
      </div>
    </div>
  );
}
