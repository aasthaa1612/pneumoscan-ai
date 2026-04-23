import React, { useState, useCallback, useEffect, useRef } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import { motion, AnimatePresence, useMotionValue, useSpring } from "framer-motion";
import {
    Upload, Activity, AlertCircle, CheckCircle, Loader, Scan, Zap,
    ChevronDown, Sparkles, Shield, TrendingUp, Bot, Send, X, RefreshCw,
    Maximize2, MessageSquare, ChevronRight, Info, FileImage, RotateCcw,
    Heart, Lungs, Stethoscope, Brain
} from "lucide-react";
import { RadialBarChart, RadialBar, PolarAngleAxis, ResponsiveContainer } from "recharts";
import "./App.css";

const API = "http://localhost:5000";

/* ── Particle Canvas ── */
function ParticleCanvas() {
    const canvasRef = useRef(null);
    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        let animId;
        const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; };
        resize();
        window.addEventListener("resize", resize);
        const particles = Array.from({ length: 60 }, () => ({
            x: Math.random() * window.innerWidth,
            y: Math.random() * window.innerHeight,
            r: Math.random() * 1.5 + 0.4,
            vx: (Math.random() - 0.5) * 0.3,
            vy: (Math.random() - 0.5) * 0.3,
            alpha: Math.random() * 0.35 + 0.1,
            pulse: Math.random() * Math.PI * 2,
        }));
        const draw = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            particles.forEach(p => { p.pulse += 0.018; });
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 120) {
                        ctx.beginPath();
                        ctx.strokeStyle = `rgba(120,160,255,${0.10 * (1 - dist / 120)})`;
                        ctx.lineWidth = 0.5;
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.stroke();
                    }
                }
            }
            particles.forEach(p => {
                const pulsedAlpha = p.alpha * (0.7 + 0.3 * Math.sin(p.pulse));
                ctx.beginPath();
                const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 3);
                grad.addColorStop(0, `rgba(160,200,255,${pulsedAlpha})`);
                grad.addColorStop(1, `rgba(79,142,255,0)`);
                ctx.fillStyle = grad;
                ctx.arc(p.x, p.y, p.r * 3, 0, Math.PI * 2);
                ctx.fill();
                p.x += p.vx; p.y += p.vy;
                if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
                if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
            });
            animId = requestAnimationFrame(draw);
        };
        draw();
        return () => { cancelAnimationFrame(animId); window.removeEventListener("resize", resize); };
    }, []);
    return <canvas ref={canvasRef} className="particle-canvas" />;
}

/* ── Mouse Glow ── */
function MouseGlow() {
    const rawX = useMotionValue(-400), rawY = useMotionValue(-400);
    const x = useSpring(rawX, { stiffness: 60, damping: 20 });
    const y = useSpring(rawY, { stiffness: 60, damping: 20 });
    useEffect(() => {
        const move = (e) => { rawX.set(e.clientX); rawY.set(e.clientY); };
        window.addEventListener("mousemove", move);
        return () => window.removeEventListener("mousemove", move);
    }, [rawX, rawY]);
    return <motion.div className="mouse-glow" style={{ left: x, top: y, translateX: "-50%", translateY: "-50%" }} />;
}

/* ── Heartbeat ECG line ── */
function HeartbeatLine({ color = "#5a96ff", active = false }) {
    return (
        <svg className="ecg-line" viewBox="0 0 300 50" preserveAspectRatio="none">
            <motion.polyline
                points="0,25 30,25 45,25 55,5 65,45 75,10 85,25 120,25 150,25 165,25 175,5 185,45 195,10 205,25 240,25 300,25"
                fill="none"
                stroke={color}
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: active ? 1 : 0, opacity: active ? 1 : 0 }}
                transition={{ duration: 1.4, ease: "easeInOut", repeat: active ? Infinity : 0, repeatDelay: 1.2 }}
            />
        </svg>
    );
}

/* ── Confidence Gauge ── */
function ConfidenceGauge({ value, label, color }) {
    const data = [{ name: label, value, fill: color }];
    return (
        <motion.div className="gauge-wrap"
            initial={{ scale: 0.7, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: "spring", stiffness: 160, damping: 14 }}>
            <ResponsiveContainer width={180} height={180}>
                <RadialBarChart cx="50%" cy="50%" innerRadius="62%" outerRadius="88%" startAngle={90} endAngle={-270} data={data}>
                    <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
                    <RadialBar dataKey="value" cornerRadius={10} background={{ fill: "rgba(255,255,255,0.03)" }} />
                </RadialBarChart>
            </ResponsiveContainer>
            <div className="gauge-label">
                <span className="gauge-val" style={{ color }}>{value.toFixed(1)}%</span>
                <span className="gauge-name">{label}</span>
            </div>
        </motion.div>
    );
}

/* ── Count Up ── */
function CountUp({ end, duration = 1200, suffix = "" }) {
    const [val, setVal] = useState(0);
    useEffect(() => {
        let start = 0;
        const step = end / (duration / 16);
        const timer = setInterval(() => {
            start = Math.min(start + step, end);
            setVal(Math.round(start));
            if (start >= end) clearInterval(timer);
        }, 16);
        return () => clearInterval(timer);
    }, [end, duration]);
    return <>{val}{suffix}</>;
}

/* ── Pulse Dots ── */
function PulseDots() {
    return (
        <div className="pulse-dots">
            {[0, 0.18, 0.36].map((delay, i) => (
                <motion.span key={i} className="pulse-dot"
                    animate={{ scale: [1, 1.6, 1], opacity: [0.4, 1, 0.4] }}
                    transition={{ duration: 0.9, delay, repeat: Infinity }} />
            ))}
        </div>
    );
}

/* ── Typewriter ── */
function TypeWriter({ text, speed = 28, onDone }) {
    const [displayed, setDisplayed] = useState("");
    const [done, setDone] = useState(false);
    useEffect(() => {
        setDisplayed("");
        setDone(false);
        let i = 0;
        const timer = setInterval(() => {
            i++;
            setDisplayed(text.slice(0, i));
            if (i >= text.length) { clearInterval(timer); setDone(true); onDone?.(); }
        }, speed);
        return () => clearInterval(timer);
    }, [text]);
    return <span>{displayed}{!done && <span className="cursor-blink">|</span>}</span>;
}

/* ── AI Chatbot ── */
const BOT_SYSTEM = `You are PneumoScan AI, a specialized radiology assistant for chest X-ray pneumonia detection. 
You help users understand their results, explain medical terms, and answer questions about pneumonia.
Be warm, clear, and always remind users to consult a real doctor for medical decisions.
Keep responses concise (2-4 sentences usually). Use simple language.`;

const QUICK_PROMPTS = [
    "What is pneumonia?",
    "How accurate is this AI?",
    "What does Grad-CAM show?",
    "Should I see a doctor?",
    "What causes pneumonia?",
    "How is it treated?",
];

function AIChatbot({ isOpen, onClose, result }) {
    const [messages, setMessages] = useState([
        {
            role: "assistant",
            content: result
                ? `I've analyzed your chest X-ray. The result shows **${result.prediction}** with ${result.confidence}% confidence. ${result.prediction === "PNEUMONIA" ? "I strongly recommend consulting a physician promptly. Can I explain what this means or answer any questions?" : "That's encouraging! No signs of pneumonia were detected. Do you have any questions about your results?"}`
                : "Hello! I'm PneumoScan AI. I can help explain pneumonia, how this detection works, or answer questions about your results. What would you like to know?",
        }
    ]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const scrollRef = useRef(null);

    useEffect(() => {
        scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
    }, [messages]);

    const sendMessage = async (text) => {
        const userText = text || input.trim();
        if (!userText) return;
        setInput("");
        const newMsgs = [...messages, { role: "user", content: userText }];
        setMessages(newMsgs);
        setLoading(true);
        try {
            const contextMsg = result
                ? `[Context: Patient X-ray shows ${result.prediction} with ${result.confidence}% confidence. Normal probability: ${result.probabilities?.NORMAL?.toFixed(1)}%, Pneumonia probability: ${result.probabilities?.PNEUMONIA?.toFixed(1)}%]\n\n${userText}`
                : userText;
            const apiMessages = [
                ...newMsgs.slice(0, -1).map(m => ({ role: m.role, content: m.content })),
                { role: "user", content: contextMsg }
            ];
            const res = await fetch(`${API}/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model: "claude-3-5-sonnet-20241022",
                    max_tokens: 1000,
                    system: BOT_SYSTEM,
                    messages: apiMessages,
                }),
            });
            const data = await res.json();

            if (!res.ok) {
                throw new Error(data.error || "Connection error");
            }

            const reply = data.content?.[0]?.text || "I'm having trouble connecting right now. Please try again.";
            setMessages(prev => [...prev, { role: "assistant", content: reply }]);
        } catch (err) {
            setMessages(prev => [...prev, { role: "assistant", content: err.message || "Connection error. Please check your backend and try again." }]);
        }
        setLoading(false);
    };

    const formatMsg = (text) => {
        return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    };

    return (
        <AnimatePresence>
            {isOpen && (
                <motion.div className="chatbot-panel"
                    initial={{ opacity: 0, x: 400, scale: 0.9 }}
                    animate={{ opacity: 1, x: 0, scale: 1 }}
                    exit={{ opacity: 0, x: 400, scale: 0.9 }}
                    transition={{ type: "spring", stiffness: 280, damping: 28 }}>

                    {/* Bot header */}
                    <div className="bot-header">
                        <div className="bot-avatar-wrap">
                            <motion.div className="bot-avatar"
                                animate={{ boxShadow: ["0 0 12px rgba(90,150,255,0.3)", "0 0 28px rgba(90,150,255,0.7)", "0 0 12px rgba(90,150,255,0.3)"] }}
                                transition={{ duration: 2.2, repeat: Infinity }}>
                                <Bot size={18} color="#fff" />
                            </motion.div>
                            <div className="bot-status-dot" />
                        </div>
                        <div className="bot-header-text">
                            <span className="bot-name">PneumoScan AI</span>
                            <span className="bot-subtitle">Medical Assistant · Powered by Claude</span>
                        </div>
                        <button className="bot-close" onClick={onClose}><X size={16} /></button>
                    </div>

                    {/* Messages */}
                    <div className="bot-messages" ref={scrollRef}>
                        <AnimatePresence initial={false}>
                            {messages.map((msg, i) => (
                                <motion.div key={i}
                                    className={`bot-msg ${msg.role}`}
                                    initial={{ opacity: 0, y: 12, scale: 0.95 }}
                                    animate={{ opacity: 1, y: 0, scale: 1 }}
                                    transition={{ type: "spring", stiffness: 200, damping: 20 }}>
                                    {msg.role === "assistant" && (
                                        <div className="msg-bot-icon"><Bot size={12} /></div>
                                    )}
                                    <div className="msg-bubble"
                                        dangerouslySetInnerHTML={{ __html: formatMsg(msg.content) }} />
                                </motion.div>
                            ))}
                        </AnimatePresence>
                        {loading && (
                            <motion.div className="bot-msg assistant"
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}>
                                <div className="msg-bot-icon"><Bot size={12} /></div>
                                <div className="msg-bubble typing-bubble">
                                    <span className="typing-dot" /><span className="typing-dot" /><span className="typing-dot" />
                                </div>
                            </motion.div>
                        )}
                    </div>

                    {/* Quick prompts */}
                    {messages.length <= 2 && (
                        <div className="quick-prompts">
                            <span className="qp-label">Quick questions</span>
                            <div className="qp-chips">
                                {QUICK_PROMPTS.map((q, i) => (
                                    <motion.button key={q} className="qp-chip"
                                        onClick={() => sendMessage(q)}
                                        initial={{ opacity: 0, y: 6 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ delay: i * 0.06 }}
                                        whileHover={{ scale: 1.03 }}
                                        whileTap={{ scale: 0.97 }}>
                                        {q}
                                    </motion.button>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Input */}
                    <div className="bot-input-row">
                        <input
                            className="bot-input"
                            placeholder="Ask about pneumonia or your results…"
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            onKeyDown={e => e.key === "Enter" && !e.shiftKey && sendMessage()}
                        />
                        <motion.button className="bot-send"
                            onClick={() => sendMessage()}
                            disabled={!input.trim() || loading}
                            whileHover={{ scale: 1.08 }}
                            whileTap={{ scale: 0.93 }}>
                            <Send size={15} />
                        </motion.button>
                    </div>
                </motion.div>
            )}
        </AnimatePresence>
    );
}

/* ── Floating Bot Button ── */
function BotTrigger({ onClick, hasResult, unread }) {
    return (
        <motion.button className="bot-trigger"
            onClick={onClick}
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: "spring", stiffness: 260, damping: 18, delay: 1.2 }}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.93 }}>
            <motion.div
                animate={{ boxShadow: ["0 0 0px rgba(90,150,255,0)", "0 0 30px rgba(90,150,255,0.6)", "0 0 0px rgba(90,150,255,0)"] }}
                transition={{ duration: 2.5, repeat: Infinity }}>
                <MessageSquare size={22} color="#fff" />
            </motion.div>
            {unread && (
                <motion.span className="bot-unread"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring", stiffness: 400 }}>1</motion.span>
            )}
            {hasResult && (
                <motion.span className="bot-ping"
                    animate={{ scale: [1, 1.8, 1], opacity: [0.8, 0, 0.8] }}
                    transition={{ duration: 2, repeat: Infinity }} />
            )}
        </motion.button>
    );
}

/* ── Result Confidence Bar ── */
function ConfidenceBar({ label, value, color, delay = 0 }) {
    return (
        <div className="conf-bar-row">
            <div className="conf-bar-label">
                <span>{label}</span>
                <span style={{ color, fontFamily: "JetBrains Mono, monospace", fontSize: "0.78rem" }}>{value.toFixed(1)}%</span>
            </div>
            <div className="conf-bar-track">
                <motion.div className="conf-bar-fill"
                    style={{ background: color }}
                    initial={{ width: 0 }}
                    animate={{ width: `${value}%` }}
                    transition={{ duration: 0.9, delay, ease: [0.34, 1.56, 0.64, 1] }} />
            </div>
        </div>
    );
}

/* ── Analyze Another CTA ── */
function AnalyzeAnotherButton({ onClick }) {
    return (
        <motion.button className="btn-analyze-another"
            onClick={onClick}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6, duration: 0.5 }}
            whileHover={{ scale: 1.03, y: -2 }}
            whileTap={{ scale: 0.97 }}>
            <RefreshCw size={16} />
            <span>Analyze Another X-Ray</span>
            <span className="btn-shimmer" />
        </motion.button>
    );
}

/* ── Image Comparison Slider ── */
function CompareSlider({ original, gradcam }) {
    const [pos, setPos] = useState(50);
    const [dragging, setDragging] = useState(false);
    const containerRef = useRef(null);

    const handleMove = useCallback((clientX) => {
        if (!containerRef.current) return;
        const rect = containerRef.current.getBoundingClientRect();
        const pct = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
        setPos(pct);
    }, []);

    return (
        <div className="compare-slider-wrap" ref={containerRef}
            onMouseMove={e => dragging && handleMove(e.clientX)}
            onMouseUp={() => setDragging(false)}
            onMouseLeave={() => setDragging(false)}
            onTouchMove={e => handleMove(e.touches[0].clientX)}
            onTouchEnd={() => setDragging(false)}>
            <img src={`data:image/png;base64,${original}`} alt="Original" className="cs-base" />
            <div className="cs-overlay" style={{ width: `${pos}%` }}>
                <img src={`data:image/png;base64,${gradcam}`} alt="Grad-CAM" className="cs-top" />
            </div>
            <div className="cs-divider" style={{ left: `${pos}%` }}
                onMouseDown={e => { e.preventDefault(); setDragging(true); }}
                onTouchStart={() => setDragging(true)}>
                <div className="cs-handle">
                    <ChevronRight size={12} className="cs-arrow-r" />
                    <ChevronRight size={12} className="cs-arrow-l" style={{ transform: "rotate(180deg)" }} />
                </div>
            </div>
            <div className="cs-labels">
                <span className="cs-lbl" style={{ left: "8px" }}>Grad-CAM</span>
                <span className="cs-lbl" style={{ right: "8px" }}>Original</span>
            </div>
        </div>
    );
}

/* ── Severity Indicator ── */
function SeverityMeter({ confidence, isPneumonia }) {
    const level = isPneumonia
        ? confidence > 90 ? { label: "High Risk", color: "#ff3d5a", bars: 3 }
            : confidence > 70 ? { label: "Moderate Risk", color: "#ff9432", bars: 2 }
                : { label: "Low Risk", color: "#ffd454", bars: 1 }
        : { label: "Clear", color: "#30d47a", bars: 0 };

    return (
        <div className="severity-wrap">
            <span className="sev-label">Risk Level</span>
            <div className="sev-bars">
                {[0, 1, 2].map(i => (
                    <motion.div key={i} className="sev-bar"
                        style={{ background: i < level.bars ? level.color : "rgba(255,255,255,0.08)" }}
                        initial={{ scaleY: 0 }}
                        animate={{ scaleY: 1 }}
                        transition={{ delay: 0.5 + i * 0.12, type: "spring", stiffness: 200 }} />
                ))}
            </div>
            <span className="sev-level" style={{ color: level.color }}>{level.label}</span>
        </div>
    );
}

/* ── Main App ── */
export default function App() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [showCam, setShowCam] = useState(true);
    const [viewMode, setViewMode] = useState("split"); // "split" | "original" | "cam" | "compare"
    const [modelInfo, setModelInfo] = useState(null);
    const [scanAnim, setScanAnim] = useState(false);
    const [botOpen, setBotOpen] = useState(false);
    const [botUnread, setBotUnread] = useState(false);
    const [showAnalyzeAnother, setShowAnalyzeAnother] = useState(false);
    const resultRef = useRef(null);
    const uploadRef = useRef(null);

    useEffect(() => {
        axios.get(`${API}/model-info`).then(r => setModelInfo(r.data)).catch(() => { });
    }, []);

    // Show unread notification after result
    useEffect(() => {
        if (result && !botOpen) {
            const t = setTimeout(() => setBotUnread(true), 2000);
            return () => clearTimeout(t);
        }
    }, [result]);

    const onDrop = useCallback((accepted) => {
        if (!accepted.length) return;
        const f = accepted[0];
        setFile(f);
        setPreview(URL.createObjectURL(f));
        setResult(null);
        setError(null);
        setShowAnalyzeAnother(false);
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { "image/*": [".jpg", ".jpeg", ".png"] },
        multiple: false,
    });

    const analyse = async () => {
        if (!file) return;
        setLoading(true);
        setError(null);
        setScanAnim(true);
        try {
            const form = new FormData();
            form.append("image", file);
            const res = await axios.post(`${API}/predict`, form, {
                headers: { "Content-Type": "multipart/form-data" },
            });
            setResult(res.data);
            setShowAnalyzeAnother(true);
            setTimeout(() => {
                resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
            }, 300);
        } catch (e) {
            setError(e.response?.data?.error || "Server error — is the backend running?");
        }
        setLoading(false);
        setScanAnim(false);
    };

    const handleAnalyzeAnother = () => {
        setFile(null);
        setPreview(null);
        setResult(null);
        setError(null);
        setShowAnalyzeAnother(false);
        setBotUnread(false);
        window.scrollTo({ top: 0, behavior: "smooth" });
        setTimeout(() => uploadRef.current?.scrollIntoView({ behavior: "smooth", block: "center" }), 400);
    };

    const isPneumonia = result?.prediction === "PNEUMONIA";
    const accentHex = isPneumonia ? "#ff5b78" : "#30d47a";
    const accentGlow = isPneumonia ? "rgba(255,91,120,0.22)" : "rgba(48,212,122,0.22)";

    return (
        <div className="app">
            {/* Backgrounds */}
            <div className="cosmos">
                <div className="cosmos-orb a" /><div className="cosmos-orb b" />
                <div className="cosmos-orb c" /><div className="cosmos-orb d" />
            </div>
            <div className="aurora">
                <div className="aurora-band a" /><div className="aurora-band b" /><div className="aurora-band c" />
            </div>
            <div className="bg-grid"><div className="bg-grid-inner" /></div>
            <ParticleCanvas />
            <MouseGlow />
            <div className="floating-rings">
                <div className="ring r1" /><div className="ring r2" /><div className="ring r3" />
            </div>

            {/* Header */}
            <motion.header className="header"
                initial={{ y: -80, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}>
                <div className="header-brand">
                    <motion.div className="logo-ring" whileHover={{ scale: 1.12, rotate: 8 }} transition={{ type: "spring", stiffness: 300 }}>
                        <Activity size={20} color="#fff" />
                    </motion.div>
                    <div className="brand-text">
                        <h1>PneumoScan AI</h1>
                        <p>ResNet-50 · Grad-CAM · Chest X-Ray</p>
                    </div>
                </div>
                <div className="header-right">
                    <HeartbeatLine color="#5a96ff" active={loading} />
                    <motion.div className="model-badge"
                        animate={{ boxShadow: ["0 0 0px rgba(79,142,255,0)", "0 0 18px rgba(79,142,255,0.2)", "0 0 0px rgba(79,142,255,0)"] }}
                        transition={{ duration: 3, repeat: Infinity }}>
                        <span className="badge-dot" />
                        {modelInfo ? <span>Val AUC {modelInfo.val_auc} · {modelInfo.device?.toUpperCase()}</span> : <span>Model Ready</span>}
                    </motion.div>
                    <motion.button className="btn-chat-header"
                        onClick={() => { setBotOpen(b => !b); setBotUnread(false); }}
                        whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                        <MessageSquare size={15} />
                        <span>Ask AI</span>
                    </motion.button>
                </div>
            </motion.header>

            {/* Section 1: Upload */}
            <section className="section section-upload" ref={uploadRef}>
                <div className="section-inner">

                    <motion.div className="hero-text"
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}>
                        <div className="hero-kicker">
                            <span className="hero-kicker-dot" />
                            AI-Powered Pneumonia Detection
                        </div>
                        <h2 className="hero-title">Upload your<br /><em>chest X-ray</em></h2>
                        <p className="hero-sub">Advanced deep learning with Grad-CAM visualization — instant results with AI explanation</p>
                    </motion.div>

                    <motion.div className="upload-card"
                        initial={{ opacity: 0, y: 50 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.7, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
                        whileHover={{ y: -4, transition: { duration: 0.3 } }}>

                        <div {...getRootProps()} className={`dropzone ${isDragActive ? "active" : ""} ${preview ? "has-preview" : ""}`}>
                            <input {...getInputProps()} />
                            <AnimatePresence mode="wait">
                                {preview ? (
                                    <motion.div key="preview" className="preview-container"
                                        initial={{ opacity: 0, scale: 0.9 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        exit={{ opacity: 0, scale: 0.9 }}
                                        transition={{ type: "spring", stiffness: 200, damping: 20 }}>
                                        <img src={preview} alt="Preview" className="preview-img" />
                                        {/* Scanning grid overlay on preview */}
                                        <div className="preview-scan-grid" />
                                        <div className="preview-overlay">
                                            <div className="preview-corners">
                                                <span className="corner tl" /><span className="corner tr" />
                                                <span className="corner bl" /><span className="corner br" />
                                            </div>
                                            <div className="preview-badge">
                                                <CheckCircle size={12} />
                                                {file?.name?.length > 22 ? file.name.slice(0, 22) + "…" : file?.name}
                                            </div>
                                        </div>
                                        {/* Change image button */}
                                        <motion.div className="preview-change-hint"
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            transition={{ delay: 0.5 }}>
                                            <FileImage size={13} />
                                            Click to change image
                                        </motion.div>
                                    </motion.div>
                                ) : (
                                    <motion.div key="placeholder" className="drop-placeholder"
                                        initial={{ opacity: 0, scale: 0.95 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        exit={{ opacity: 0 }}>
                                        <motion.div className="drop-icon-wrap"
                                            animate={{ y: [0, -8, 0] }}
                                            transition={{ duration: 3.2, repeat: Infinity, ease: "easeInOut" }}>
                                            <Upload size={28} />
                                        </motion.div>
                                        <p className="drop-title">Drop your chest X-ray here</p>
                                        <p className="drop-sub">or click to browse files</p>
                                        <div className="drop-formats">
                                            {["JPG", "JPEG", "PNG"].map((fmt, i) => (
                                                <motion.span key={fmt} className="format-chip"
                                                    initial={{ opacity: 0, y: 6 }}
                                                    animate={{ opacity: 1, y: 0 }}
                                                    transition={{ delay: 0.3 + i * 0.08 }}>{fmt}</motion.span>
                                            ))}
                                        </div>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>

                        <motion.button className="btn-analyse"
                            onClick={analyse}
                            disabled={!file || loading}
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.97 }}>
                            {loading ? (
                                <><Loader size={17} className="spin" /><span>Analysing</span><PulseDots /></>
                            ) : (
                                <><Scan size={17} /><span>Analyse X-Ray</span></>
                            )}
                            <span className="btn-shimmer" />
                        </motion.button>

                        <AnimatePresence>
                            {error && (
                                <motion.div className="error-box"
                                    initial={{ opacity: 0, y: -8, height: 0 }}
                                    animate={{ opacity: 1, y: 0, height: "auto" }}
                                    exit={{ opacity: 0, y: -8, height: 0 }}>
                                    <AlertCircle size={15} /> {error}
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </motion.div>

                    {/* Stats */}
                    <motion.div className="stats-row"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.45, duration: 0.6 }}>
                        {[
                            { icon: <TrendingUp size={15} />, num: "5,216", lbl: "Training Images" },
                            { icon: <Shield size={15} />, num: "98.4%", lbl: "Test Accuracy" },
                            { icon: <Sparkles size={15} />, num: "ResNet-50", lbl: "Architecture" },
                        ].map((s, i) => (
                            <motion.div className="stat-pill" key={s.lbl}
                                initial={{ opacity: 0, y: 14 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.55 + i * 0.09 }}
                                whileHover={{ scale: 1.05, y: -3 }}>
                                <span className="stat-icon">{s.icon}</span>
                                <span className="stat-num">{s.num}</span>
                                <span className="stat-lbl">{s.lbl}</span>
                            </motion.div>
                        ))}
                    </motion.div>

                    <AnimatePresence>
                        {result && !loading && (
                            <motion.div
                                initial={{ opacity: 0, y: -10 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0 }}>
                                <div className="scroll-hint">
                                    <span>Scroll to see results</span>
                                    <motion.div animate={{ y: [0, 6, 0] }} transition={{ duration: 1.4, repeat: Infinity }}>
                                        <ChevronDown size={16} />
                                    </motion.div>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </section>

            {/* Section 2: Results */}
            <AnimatePresence>
                {(result || loading) && (
                    <section ref={resultRef} className="section section-results">
                        <div className="section-inner">
                            {loading ? (
                                <motion.div className="scan-loader"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    exit={{ opacity: 0 }}>
                                    <div className="scan-orbit-wrap">
                                        <div className="scan-orbit o1"><div className="orbit-dot" /></div>
                                        <div className="scan-orbit o2"><div className="orbit-dot" /></div>
                                        <div className="scan-orbit o3"><div className="orbit-dot" /></div>
                                        <motion.div className="scan-core"
                                            animate={{ boxShadow: ["0 0 20px rgba(79,142,255,0.4)", "0 0 50px rgba(79,142,255,0.9)", "0 0 20px rgba(79,142,255,0.4)"] }}
                                            transition={{ duration: 1.8, repeat: Infinity }}>
                                            <Activity size={24} color="#fff" />
                                        </motion.div>
                                    </div>
                                    <motion.p className="scan-label"
                                        animate={{ opacity: [0.6, 1, 0.6] }}
                                        transition={{ duration: 1.6, repeat: Infinity }}>
                                        Analysing X-ray…
                                    </motion.p>
                                    <p className="scan-sub">Running ResNet-50 inference + Grad-CAM</p>
                                    <div className="scan-steps">
                                        {["Preprocessing image", "Running ResNet-50", "Computing Grad-CAM", "Generating results"].map((step, i) => (
                                            <motion.div key={step} className="scan-step"
                                                initial={{ opacity: 0, x: -12 }}
                                                animate={{ opacity: 1, x: 0 }}
                                                transition={{ delay: i * 0.6 }}>
                                                <motion.div className="scan-step-dot"
                                                    animate={{ scale: [1, 1.4, 1], opacity: [0.5, 1, 0.5] }}
                                                    transition={{ duration: 0.8, repeat: Infinity, delay: i * 0.6 }} />
                                                <span>{step}</span>
                                            </motion.div>
                                        ))}
                                    </div>
                                    <div className="progress-track">
                                        <motion.div className="progress-fill"
                                            initial={{ width: "0%" }}
                                            animate={{ width: ["0%", "35%", "60%", "80%", "92%"] }}
                                            transition={{ duration: 3, ease: "easeOut" }} />
                                    </div>
                                </motion.div>
                            ) : result && (
                                <motion.div className="results-grid"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    transition={{ duration: 0.5 }}>

                                    {/* Verdict */}
                                    <motion.div className="result-header"
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ delay: 0.1 }}>
                                        <p className="section-eyebrow"><Scan size={12} /> Analysis Result</p>
                                        <motion.div className="verdict-banner"
                                            style={{ borderColor: accentHex, background: `linear-gradient(135deg, ${accentGlow}, transparent 80%)` }}
                                            initial={{ scale: 0.94, opacity: 0 }}
                                            animate={{ scale: 1, opacity: 1 }}
                                            transition={{ delay: 0.2, type: "spring", stiffness: 220 }}>
                                            <motion.div className="verdict-icon"
                                                style={{ background: `${accentHex}18`, borderColor: `${accentHex}50` }}
                                                animate={{ boxShadow: [`0 0 0px ${accentHex}00`, `0 0 28px ${accentHex}70`, `0 0 0px ${accentHex}00`] }}
                                                transition={{ duration: 2.2, repeat: Infinity }}>
                                                {isPneumonia ? <AlertCircle size={28} color={accentHex} /> : <CheckCircle size={28} color={accentHex} />}
                                            </motion.div>
                                            <div className="verdict-text">
                                                <span className="verdict-label" style={{ color: accentHex }}>{result.prediction}</span>
                                                <span className="verdict-conf">Confidence: {result.confidence}%</span>
                                            </div>
                                            <SeverityMeter confidence={parseFloat(result.confidence)} isPneumonia={isPneumonia} />
                                            <motion.div className="verdict-badge"
                                                style={{ background: `${accentHex}18`, color: accentHex, borderColor: `${accentHex}40` }}
                                                animate={{ scale: [1, 1.04, 1] }}
                                                transition={{ duration: 2, repeat: Infinity }}>
                                                {isPneumonia ? "⚠ Detected" : "✓ Clear"}
                                            </motion.div>
                                        </motion.div>
                                    </motion.div>

                                    {/* Probability bars + gauges */}
                                    <motion.div className="gauges-card"
                                        initial={{ opacity: 0, y: 24 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ delay: 0.25 }}>
                                        <p className="card-eyebrow">Probability Distribution</p>
                                        <div className="gauges-row">
                                            <ConfidenceGauge value={result.probabilities.NORMAL} label="Normal" color="#30d47a" />
                                            <div className="gauge-divider" />
                                            <ConfidenceGauge value={result.probabilities.PNEUMONIA} label="Pneumonia" color="#ff5b78" />
                                        </div>
                                        {/* Confidence bars */}
                                        <div className="conf-bars-section">
                                            <ConfidenceBar label="Normal" value={result.probabilities.NORMAL} color="#30d47a" delay={0.3} />
                                            <ConfidenceBar label="Pneumonia" value={result.probabilities.PNEUMONIA} color="#ff5b78" delay={0.45} />
                                        </div>
                                    </motion.div>

                                    {/* Image viewer with modes */}
                                    <motion.div className="viewer-card"
                                        initial={{ opacity: 0, y: 24 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ delay: 0.35 }}>
                                        <p className="card-eyebrow">Visual Explanation</p>

                                        <div className="img-toggle-row">
                                            {[
                                                { key: "original", label: "Original" },
                                                { key: "cam", label: "Grad-CAM" },
                                                { key: "compare", label: "Compare Slider" },
                                            ].map(btn => (
                                                <button key={btn.key}
                                                    className={`toggle-btn ${viewMode === btn.key ? "active" : ""}`}
                                                    onClick={() => setViewMode(btn.key)}>
                                                    {btn.label}
                                                </button>
                                            ))}
                                        </div>

                                        <AnimatePresence mode="wait">
                                            {viewMode === "compare" ? (
                                                <motion.div key="compare"
                                                    initial={{ opacity: 0 }}
                                                    animate={{ opacity: 1 }}
                                                    exit={{ opacity: 0 }}>
                                                    <CompareSlider original={result.original_image} gradcam={result.gradcam_overlay} />
                                                    <p className="compare-hint">← Drag slider to compare →</p>
                                                </motion.div>
                                            ) : (
                                                <motion.div key={viewMode} className="cam-img-wrap"
                                                    initial={{ opacity: 0, filter: "blur(10px)" }}
                                                    animate={{ opacity: 1, filter: "blur(0px)" }}
                                                    exit={{ opacity: 0, filter: "blur(10px)" }}
                                                    transition={{ duration: 0.4 }}>
                                                    <img
                                                        src={`data:image/png;base64,${viewMode === "cam" ? result.gradcam_overlay : result.original_image}`}
                                                        alt={viewMode === "cam" ? "Grad-CAM" : "Original"}
                                                        className="cam-img"
                                                    />
                                                    {scanAnim && <div className="scan-line" />}
                                                    <div className="cam-corners">
                                                        <span className="c tl" /><span className="c tr" />
                                                        <span className="c bl" /><span className="c br" />
                                                    </div>
                                                    {viewMode === "cam" && (
                                                        <motion.div className="cam-legend"
                                                            initial={{ opacity: 0, y: 6 }}
                                                            animate={{ opacity: 1, y: 0 }}
                                                            transition={{ delay: 0.3 }}>
                                                            <span className="legend-item hot"><span className="legend-dot" /> High activation</span>
                                                            <span className="legend-item cool"><span className="legend-dot" /> Low activation</span>
                                                        </motion.div>
                                                    )}
                                                </motion.div>
                                            )}
                                        </AnimatePresence>
                                        <p className="disclaimer">⚠ For research purposes only · Not a substitute for clinical diagnosis</p>
                                    </motion.div>

                                    {/* Analyze Another */}
                                    <AnalyzeAnotherButton onClick={handleAnalyzeAnother} />

                                    {/* Ask AI nudge */}
                                    <AnimatePresence>
                                        {!botOpen && (
                                            <motion.div className="bot-nudge"
                                                initial={{ opacity: 0, y: 10 }}
                                                animate={{ opacity: 1, y: 0 }}
                                                exit={{ opacity: 0 }}
                                                transition={{ delay: 1 }}>
                                                <Bot size={15} color="#5a96ff" />
                                                <span>Have questions about these results?</span>
                                                <button onClick={() => { setBotOpen(true); setBotUnread(false); }}>
                                                    Ask AI Assistant <ChevronRight size={13} />
                                                </button>
                                            </motion.div>
                                        )}
                                    </AnimatePresence>
                                </motion.div>
                            )}
                        </div>
                    </section>
                )}
            </AnimatePresence>

            {/* Footer */}
            <motion.footer className="footer"
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1, duration: 0.5 }}>
                <div className="footer-inner">
                    <span className="footer-text">Made</span>
                    <span className="footer-text">by</span>
                    <span className="footer-name">Aastha</span>
                    <span className="footer-amp">&amp;</span>
                    <span className="footer-name">Kritika</span>
                </div>
            </motion.footer>

            {/* Floating bot button */}
            <BotTrigger
                onClick={() => { setBotOpen(b => !b); setBotUnread(false); }}
                hasResult={!!result}
                unread={botUnread}
            />

            {/* Chatbot panel */}
            <AIChatbot isOpen={botOpen} onClose={() => setBotOpen(false)} result={result} />
        </div>
    );
}