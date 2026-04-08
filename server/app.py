"""
FastAPI application exposing the SupportEnv via the standard OpenEnv API:
  POST /reset   -> resets the environment, returns initial Observation
  POST /step    -> executes one Action, returns StepResult
  GET  /state   -> returns current State (episode metadata)
  GET  /health  -> health check
  GET  /docs    -> OpenAPI Swagger UI (automatic)
"""

import os
import time
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

from server.support_env import SupportEnv
from support_env.models import Action, Observation, Reward

# --- App ----------------------------------------------------------------
app = FastAPI(
    title="Support Ticket Environment",
    description=(
        "An RL environment where an AI agent triages, classifies, responds to, "
        "escalates, and resolves customer support tickets under SLA pressure."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Shared environment instance ------------------------------------------
_env = SupportEnv()
_episode_id: str = str(uuid.uuid4())
_episode_start: float = time.time()


# --- Response models ----------------------------------------------------

class StepResult(BaseModel):
    observation: Observation
    reward: float
    reward_breakdown: dict
    customer_satisfaction_impact: float
    team_efficiency_impact: float
    long_term_value_impact: float
    done: bool
    info: dict


class State(BaseModel):
    episode_id: str
    step_count: int
    resolved_count: int
    open_count: int
    total_tickets: int
    elapsed_seconds: float
    history: list


class HealthResponse(BaseModel):
    status: str
    version: str


# --- Endpoints ---------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    """Health check -- must return 200 for HF Spaces ping."""
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/reset", response_model=Observation, tags=["Environment"])
def reset():
    """Reset the environment, returning the initial observation."""
    global _episode_id, _episode_start
    _episode_id = str(uuid.uuid4())
    _episode_start = time.time()
    obs = _env.reset()
    return obs


@app.post("/step", response_model=StepResult, tags=["Environment"])
def step(action: Action):
    """Execute one action and return the resulting StepResult."""
    obs, reward, done, info = _env.step(action)
    return StepResult(
        observation=obs,
        reward=reward.score,
        reward_breakdown=reward.breakdown,
        customer_satisfaction_impact=getattr(reward, 'customer_satisfaction_impact', 0.0),
        team_efficiency_impact=getattr(reward, 'team_efficiency_impact', 0.0),
        long_term_value_impact=getattr(reward, 'long_term_value_impact', 0.0),
        done=done,
        info=info,
    )


@app.get("/state", response_model=State, tags=["Environment"])
def state():
    """Return the current episode state (metadata)."""
    tickets = _env.tickets if _env.tickets else []
    return State(
        episode_id=_episode_id,
        step_count=_env.step_count,
        resolved_count=sum(1 for t in tickets if t.resolved),
        open_count=sum(1 for t in tickets if not t.resolved),
        total_tickets=_env.ticket_count,
        elapsed_seconds=round(time.time() - _episode_start, 2),
        history=_env.history,
    )


# --- Web UI ------------------------------------------------------------

WEB_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Support Ticket RL Environment | Enterprise Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
<style>
  :root{--bg:#0a0b0f;--surface:#151823;--surface-light:#1a1d29;--accent:#6366f1;--accent-light:#818cf8;
        --green:#10b981;--green-light:#34d399;--red:#ef4444;--red-light:#f87171;--yellow:#f59e0b;
        --text:#f8fafc;--text-muted:#94a3b8;--border:#2d3748;--border-light:#374151;
        --gradient-1:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
        --gradient-2:linear-gradient(135deg,#f093fb 0%,#f5576c 100%);
        --gradient-3:linear-gradient(135deg,#4facfe 0%,#00f2fe 100%);
        --gradient-accent:linear-gradient(135deg,var(--accent),#a855f7);
        --gradient-success:linear-gradient(135deg,var(--green),#059669);
        --shadow-sm:0 1px 2px 0 rgba(0,0,0,0.05);
        --shadow-md:0 4px 6px -1px rgba(0,0,0,0.1),0 2px 4px -1px rgba(0,0,0,0.06);
        --shadow-lg:0 10px 15px -3px rgba(0,0,0,0.1),0 4px 6px -2px rgba(0,0,0,0.05);
        --shadow-xl:0 20px 25px -5px rgba(0,0,0,0.1),0 10px 10px -5px rgba(0,0,0,0.04);
        --neon-glow:0 0 20px rgba(99,102,241,0.5),0 0 40px rgba(99,102,241,0.3);
        --neon-green:0 0 20px rgba(16,185,129,0.5),0 0 40px rgba(16,185,129,0.3)}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;
       min-height:100vh;padding:1.5rem;line-height:1.6;overflow-x:hidden;position:relative}
  
  /* Animated Background Particles */
  .bg-animation{position:fixed;top:0;left:0;width:100%;height:100%;z-index:-1;overflow:hidden}
  .bg-animation .orb{position:absolute;border-radius:50%;filter:blur(100px);opacity:0.3;animation:float 20s infinite}
  .bg-animation .orb:nth-child(1){width:400px;height:400px;background:var(--accent);top:10%;left:10%;animation-delay:0s}
  .bg-animation .orb:nth-child(2){width:300px;height:300px;background:var(--green);top:60%;right:10%;animation-delay:-5s}
  .bg-animation .orb:nth-child(3){width:350px;height:350px;background:#a855f7;top:40%;left:60%;animation-delay:-10s}
  @keyframes float{0%,100%{transform:translate(0,0) scale(1)}25%{transform:translate(50px,-50px) scale(1.1)}50%{transform:translate(-30px,30px) scale(0.9)}75%{transform:translate(20px,20px) scale(1.05)}}
  
  /* Glass Morphism */
  .glass{background:rgba(21,24,35,0.7);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px)}
  .container{max-width:1400px;margin:0 auto;position:relative;z-index:1}
  .header{text-align:center;margin-bottom:2.5rem;position:relative;animation:fadeInDown 0.8s ease}
  @keyframes fadeInDown{from{opacity:0;transform:translateY(-30px)}to{opacity:1;transform:translateY(0)}}
  .header::before{content:'';position:absolute;top:-20px;left:50%;transform:translateX(-50%);
                  width:200px;height:2px;background:var(--gradient-accent);border-radius:2px;
                  animation:expandWidth 1s ease-out}
  @keyframes expandWidth{from{width:0}to{width:200px}}
  h1{font-size:2.5rem;font-weight:800;background:var(--gradient-accent);
     -webkit-background-clip:text;-webkit-text-fill-color:transparent;
     margin-bottom:0.5rem;letter-spacing:-0.02em;text-shadow:0 0 30px rgba(99,102,241,0.3)}
  .sub{color:var(--text-muted);font-size:1rem;font-weight:400;margin-bottom:1rem;animation:fadeIn 1s ease 0.3s both}
  @keyframes fadeIn{from{opacity:0}to{opacity:1}}
  .badge-header{display:inline-block;background:var(--gradient-3);color:var(--bg);
                padding:0.25rem 0.75rem;border-radius:9999px;font-size:0.75rem;
                font-weight:600;margin-top:0.5rem;animation:pulse-badge 2s infinite}
  @keyframes pulse-badge{0%,100%{box-shadow:0 0 0 0 rgba(79,172,254,0.4)}50%{box-shadow:0 0 0 10px rgba(79,172,254,0)}}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:1.5rem;margin-bottom:1.5rem}
  .card{background:var(--surface);border-radius:16px;padding:1.5rem;
        border:1px solid var(--border);position:relative;overflow:hidden;
        transition:all 0.4s cubic-bezier(0.175,0.885,0.32,1.275);box-shadow:var(--shadow-md);
        animation:fadeInUp 0.6s ease;transform-style:preserve-3d;perspective:1000px}
  @keyframes fadeInUp{from{opacity:0;transform:translateY(30px)}to{opacity:1;transform:translateY(0)}}
  .card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
               background:var(--gradient-accent);opacity:0.8;animation:shimmer-border 3s infinite}
  @keyframes shimmer-border{0%{background-position:-200% 0}100%{background-position:200% 0}}
  .card:hover{transform:translateY(-5px) rotateX(5deg);box-shadow:var(--neon-glow);border-color:var(--accent-light)}
  .card h2{font-size:1rem;font-weight:600;color:var(--text);
           text-transform:uppercase;letter-spacing:.05em;margin-bottom:1.25rem;
           display:flex;align-items:center;gap:0.5rem}
  .card h2 i{color:var(--accent);font-size:0.9rem;animation:icon-bounce 2s infinite}
  @keyframes icon-bounce{0%,100%{transform:translateY(0)}50%{transform:translateY(-3px)}}
  .metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(100px,1fr));gap:1rem}
  .metric{background:var(--surface-light);border-radius:12px;padding:1rem;text-align:center;
          border:1px solid var(--border);transition:all 0.3s ease;position:relative;overflow:hidden}
  .metric::before{content:'';position:absolute;top:0;left:-100%;width:100%;height:100%;
                 background:linear-gradient(90deg,transparent,rgba(99,102,241,0.1),transparent);
                 transition:left 0.5s}
  .metric:hover::before{left:100%}
  .metric:hover{transform:scale(1.05);border-color:var(--accent);box-shadow:var(--neon-glow)}
  .metric .val{font-size:2rem;font-weight:800;color:var(--accent);line-height:1;animation:countUp 0.5s ease}
  @keyframes countUp{from{transform:scale(0)}to{transform:scale(1)}}
  .metric .lbl{font-size:0.75rem;color:var(--text-muted);margin-top:0.25rem;text-transform:uppercase;letter-spacing:0.05em}
  .btn{background:var(--gradient-accent);color:#fff;border:none;border-radius:10px;
       padding:0.75rem 1.5rem;font-weight:600;cursor:pointer;font-size:0.875rem;
       transition:all 0.3s ease;position:relative;overflow:hidden;
       text-transform:uppercase;letter-spacing:0.05em;box-shadow:var(--shadow-md);
       animation:glow-pulse 2s infinite}
  @keyframes glow-pulse{0%,100%{box-shadow:0 0 5px rgba(99,102,241,0.5),0 0 20px rgba(99,102,241,0.3)}50%{box-shadow:0 0 20px rgba(99,102,241,0.8),0 0 40px rgba(99,102,241,0.5)}}
  .btn::before{content:'';position:absolute;top:0;left:-100%;width:100%;height:100%;
               background:linear-gradient(90deg,transparent,rgba(255,255,255,0.3),transparent);
               transition:left 0.6s}
  .btn:hover::before{left:100%}
  .btn:hover{transform:translateY(-3px) scale(1.02);box-shadow:var(--neon-glow)}
  .btn:active{transform:translateY(-1px) scale(0.98)}
  .btn.reset{background:var(--gradient-success);animation:glow-pulse-green 2s infinite}
  @keyframes glow-pulse-green{0%,100%{box-shadow:0 0 5px rgba(16,185,129,0.5),0 0 20px rgba(16,185,129,0.3)}50%{box-shadow:0 0 20px rgba(16,185,129,0.8),0 0 40px rgba(16,185,129,0.5)}}
  .btn-group{display:flex;gap:1rem;margin-top:1rem}
  .btn-group .btn{flex:1}
  label{display:block;font-size:0.875rem;color:var(--text-muted);margin-bottom:0.5rem;
         font-weight:500;text-transform:uppercase;letter-spacing:0.05em}
  select,input,textarea{width:100%;background:var(--surface-light);border:1px solid var(--border);
                        border-radius:8px;padding:0.75rem 1rem;color:var(--text);
                        font-size:0.875rem;margin-bottom:1rem;transition:all 0.3s ease;
                        font-family:'Inter',system-ui,sans-serif}
  select:focus,input:focus,textarea:focus{outline:none;border-color:var(--accent);
                                         box-shadow:0 0 0 3px rgba(99,102,241,0.1)}
  textarea{resize:vertical;min-height:80px}
  .log{background:var(--surface-light);border-radius:12px;padding:1rem;
       font-family:'JetBrains Mono',monospace;font-size:0.8rem;
       max-height:400px;overflow-y:auto;border:1px solid var(--border);
       box-shadow:inset 0 2px 4px rgba(0,0,0,0.1)}
  .log .entry{padding:0.75rem;border-bottom:1px solid var(--border);
              transition:background 0.2s ease}
  .log .entry:hover{background:var(--surface)}
  .log .entry:last-child{border:none}
  .tag{display:inline-flex;align-items:center;border-radius:6px;padding:0.25rem 0.75rem;
       font-size:0.7rem;font-weight:600;margin-right:0.5rem;text-transform:uppercase;
       letter-spacing:0.05em;box-shadow:var(--shadow-sm)}
  .tag i{margin-right:0.25rem;font-size:0.6rem}
  .tag.classify{background:rgba(59,130,246,0.2);color:#60a5fa;border:1px solid rgba(59,130,246,0.3)}
  .tag.respond{background:rgba(16,185,129,0.2);color:#34d399;border:1px solid rgba(16,185,129,0.3)}
  .tag.escalate{background:rgba(245,158,11,0.2);color:#fbbf24;border:1px solid rgba(245,158,11,0.3)}
  .tag.resolve{background:rgba(168,85,247,0.2);color:#c084fc;border:1px solid rgba(168,85,247,0.3)}
  .tag.transfer{background:rgba(139,92,246,0.2);color:#c4b5fd;border:1px solid rgba(139,92,246,0.3)}
  .tag.schedule_callback{background:rgba(236,72,153,0.2);color:#f472b6;border:1px solid rgba(236,72,153,0.3)}
  .tag.request_supervisor{background:rgba(249,115,22,0.2);color:#fb923c;border:1px solid rgba(249,115,22,0.3)}
  .badge{display:inline-flex;align-items:center;border-radius:9999px;padding:0.25rem 0.75rem;
         font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em}
  .badge.high,.badge.urgent{background:rgba(239,68,68,0.2);color:#f87171;border:1px solid rgba(239,68,68,0.3)}
  .badge.medium{background:rgba(245,158,11,0.2);color:#fbbf24;border:1px solid rgba(245,158,11,0.3)}
  .badge.low{background:rgba(16,185,129,0.2);color:#34d399;border:1px solid rgba(16,185,129,0.3)}
  .ticket-row{display:flex;align-items:flex-start;gap:1rem;padding:1rem 0;
              border-bottom:1px solid var(--border);transition:all 0.3s ease;
              animation:slideIn 0.5s ease both;opacity:0;transform:translateX(-20px)}
  .ticket-row:nth-child(1){animation-delay:0.1s}
  .ticket-row:nth-child(2){animation-delay:0.2s}
  .ticket-row:nth-child(3){animation-delay:0.3s}
  .ticket-row:nth-child(4){animation-delay:0.4s}
  .ticket-row:nth-child(5){animation-delay:0.5s}
  .ticket-row:nth-child(6){animation-delay:0.6s}
  @keyframes slideIn{to{opacity:1;transform:translateX(0)}}
  .ticket-row:hover{background:var(--surface-light);margin:0 -1rem;padding:1rem;
                   border-radius:8px;box-shadow:var(--shadow-md);transform:translateX(10px)}
  .ticket-row:last-child{border:none}
  .tic-id{font-weight:700;color:var(--accent);min-width:3rem;font-size:1.1rem;
          display:flex;align-items:center;gap:0.5rem;position:relative}
  .tic-id::after{content:'';position:absolute;bottom:-2px;left:0;width:0;height:2px;
                background:var(--gradient-accent);transition:width 0.3s ease}
  .ticket-row:hover .tic-id::after{width:100%}
  .tic-id i{color:var(--text-muted);font-size:0.8rem;transition:all 0.3s ease}
  .ticket-row:hover .tic-id i{color:var(--accent);transform:rotate(15deg) scale(1.2)}
  .tic-content{flex:1}
  .tic-text{font-size:0.9rem;color:var(--text);margin-bottom:0.5rem;line-height:1.5;
           transition:all 0.3s ease}
  .ticket-row:hover .tic-text{color:var(--accent-light)}
  .tic-meta{display:flex;flex-wrap:wrap;gap:0.5rem;align-items:center;
            font-size:0.75rem;color:var(--text-muted)}
  .resolved-badge{display:inline-flex;align-items:center;gap:0.25rem;
                  color:var(--green);font-size:0.75rem;font-weight:600;
                  background:rgba(16,185,129,0.1);padding:0.25rem 0.5rem;border-radius:6px;
                  animation:pulse-resolve 2s infinite}
  @keyframes pulse-resolve{0%,100%{opacity:1}50%{opacity:0.7}}
  .progress-container{margin-top:1.5rem}
  .progress-label{font-size:0.875rem;color:var(--text-muted);margin-bottom:0.5rem;
                 display:flex;justify-content:space-between;align-items:center}
  .progress-bar{height:8px;border-radius:9999px;background:var(--surface-light);
                overflow:hidden;position:relative;box-shadow:inset 0 2px 4px rgba(0,0,0,0.1)}
  .progress-fill{height:100%;border-radius:9999px;
                 background:var(--gradient-success);transition:width 0.6s ease;
                 position:relative;overflow:hidden}
  .progress-fill::after{content:'';position:absolute;top:0;left:0;right:0;bottom:0;
                        background:linear-gradient(90deg,transparent,rgba(255,255,255,0.3),transparent);
                        animation:shimmer 2s infinite}
  @keyframes shimmer{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}
  .status-indicator{display:inline-block;width:8px;height:8px;border-radius:50%;
                    margin-right:0.5rem;animation:pulse 2s infinite}
  .status-indicator.active{background:var(--green)}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.5}}
  .empty-state{text-align:center;color:var(--text-muted);padding:2rem;
               font-style:italic}
  .stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1rem;
             margin-top:1rem}
  .stat-item{text-align:center;padding:1rem;background:var(--surface-light);border-radius:8px;
            border:1px solid var(--border)}
  .stat-value{font-size:1.5rem;font-weight:700;color:var(--accent)}
  .stat-label{font-size:0.7rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em}
  @media(max-width:768px){
    .grid{grid-template-columns:1fr}
    h1{font-size:2rem}
    .metrics{grid-template-columns:repeat(2,1fr)}
    .btn-group{flex-direction:column}
  }
</style>
</head>
<body>
<div class="bg-animation">
  <div class="orb"></div>
  <div class="orb"></div>
  <div class="orb"></div>
</div>
<div class="container">
  <div class="header">
    <h1><i class="fas fa-headset"></i> Support Ticket RL Environment</h1>
    <p class="sub">Enterprise AI Agent Training Platform &mdash; Real-Time Decision Making Under Pressure</p>
    <div class="badge-header"><i class="fas fa-bolt"></i> Meta × Scaler OpenEnv Hackathon 2026</div>
  </div>

  <div class="grid">
  <!-- Metrics -->
  <div class="card">
    <h2><i class="fas fa-chart-line"></i> Current State</h2>
    <div class="metrics">
      <div class="metric"><div class="val" id="m-step">-</div><div class="lbl">Steps</div></div>
      <div class="metric"><div class="val" id="m-res">-</div><div class="lbl">Resolved</div></div>
      <div class="metric"><div class="val" id="m-open">-</div><div class="lbl">Open</div></div>
      <div class="metric"><div class="val" id="m-reward" style="color:var(--green)">-</div><div class="lbl">Last Reward</div></div>
    </div>
    <div class="progress-container">
      <div class="progress-label">
        <span><i class="fas fa-tasks"></i> Resolution Progress</span>
        <span id="progress-text">0%</span>
      </div>
      <div class="progress-bar"><div class="progress-fill" id="progress-bar" style="width:0%"></div></div>
    </div>
  </div>

  <!-- Controls -->
  <div class="card">
    <h2><i class="fas fa-gamepad"></i> Agent Control Panel</h2>
    <label for="action-type"><i class="fas fa-play-circle"></i> Action Type</label>
    <select id="action-type" onchange="updateFormFields()">
      <option value="classify">Classify Ticket</option>
      <option value="respond">Respond to Customer</option>
      <option value="resolve">Resolve & Close</option>
      <option value="escalate">Escalate to Specialist</option>
      <option value="transfer">Transfer to Agent</option>
      <option value="schedule_callback">Schedule Callback</option>
      <option value="request_supervisor">Request Supervisor</option>
    </select>
    <label for="ticket-id"><i class="fas fa-ticket"></i> Ticket ID</label>
    <input type="number" id="ticket-id" value="1" min="1" placeholder="Enter ticket ID..."/>
    
    <div id="field-classify" style="display:none">
      <label for="category"><i class="fas fa-tags"></i> Category</label>
      <select id="category" onchange="populateSubcategories()">
        <option value="">Select category...</option>
        <option value="billing">Billing Issue</option>
        <option value="technical">Technical Problem</option>
        <option value="account">Account Management</option>
        <option value="enterprise">Enterprise Support</option>
      </select>
      <label for="subcategory"><i class="fas fa-layer-group"></i> Subcategory</label>
      <select id="subcategory">
        <option value="">Select subcategory...</option>
      </select>
    </div>
    
    <div id="field-respond" style="display:none">
      <label for="response-text"><i class="fas fa-comment-dots"></i> Response Text</label>
      <textarea id="response-text" placeholder="Enter your customer response here..."></textarea>
    </div>
    
    <div id="field-escalate" style="display:none">
      <label for="escalate-to"><i class="fas fa-arrow-up"></i> Escalate To</label>
      <select id="escalate-to">
        <option value="">Select department...</option>
        <option value="senior_support">Senior Support</option>
        <option value="specialist">Specialist Team</option>
        <option value="supervisor">Supervisor</option>
        <option value="management">Management</option>
      </select>
    </div>
    
    <div id="field-transfer" style="display:none">
      <label for="transfer-to"><i class="fas fa-exchange-alt"></i> Transfer To Agent</label>
      <select id="transfer-to">
        <option value="">Select agent...</option>
        <option value="agent_1">Sarah Johnson (Billing/Account)</option>
        <option value="agent_2">Mike Chen (Technical)</option>
        <option value="agent_3">Elena Rodriguez (Enterprise)</option>
      </select>
    </div>
    

    <div id="field-callback" style="display:none">
      <label for="callback-time"><i class="fas fa-calendar-clock"></i> Callback Time (optional)</label>
      <input type="text" id="callback-time" placeholder="e.g., '2 hours', 'tomorrow'"/>
    </div>
    <div class="btn-group">
      <button class="btn" onclick="doStep()"><i class="fas fa-play"></i> STEP</button>
      <button class="btn reset" onclick="doReset()"><i class="fas fa-redo"></i> RESET</button>
    </div>
  </div>
</div>

<div class="grid">
  <!-- Tickets -->
  <div class="card">
    <h2><i class="fas fa-clipboard-list"></i> Active Tickets</h2>
    <div id="ticket-list"><div class="empty-state"><i class="fas fa-inbox"></i><br>Press "RESET" to start training...</div></div>
  </div>

  <!-- Log -->
  <div class="card">
    <h2><i class="fas fa-history"></i> Action History</h2>
    <div class="log" id="log-box"><div class="empty-state"><i class="fas fa-terminal"></i><br>No actions recorded yet...</div></div>
  </div>
</div>

<script>
const BASE = "";
let cumReward = 0;
let totalTickets = 6;

async function api(path, method="GET", body=null){
  const opts={method,headers:{"Content-Type":"application/json"}};
  if(body) opts.body=JSON.stringify(body);
  const r=await fetch(BASE+path,opts);
  return r.json();
}

function setPriority(p){
  const normalized = (p || '').toLowerCase();
  const icons = {
    high: 'exclamation-triangle',
    urgent: 'exclamation-triangle',
    medium: 'minus-circle',
    low: 'check-circle'
  };
  const label = p || '';
  return `<span class="badge ${normalized}"><i class="fas fa-${icons[normalized] || 'circle'}"></i> ${label}</span>`;
}

function updateFormFields(){
  const actionType = document.getElementById("action-type").value;
  const fields = ["classify", "respond", "escalate", "transfer", "callback"];
  fields.forEach(f => {
    const el = document.getElementById("field-" + f);
    if(el) el.style.display = "none";
  });
  
  if(actionType === "classify") {
    document.getElementById("field-classify").style.display = "block";
    populateSubcategories();
  }
  if(actionType === "respond") document.getElementById("field-respond").style.display = "block";
  if(actionType === "escalate") document.getElementById("field-escalate").style.display = "block";
  if(actionType === "transfer") document.getElementById("field-transfer").style.display = "block";
  if(actionType === "schedule_callback") document.getElementById("field-callback").style.display = "block";
}

function populateSubcategories(){
  const category = document.getElementById("category").value;
  const subcategory = document.getElementById("subcategory");
  const options = {
    billing: [
      {value: 'invoice_error', label: 'Invoice Error'},
      {value: 'refund_request', label: 'Refund Request'},
      {value: 'payment_failure', label: 'Payment Failure'},
    ],
    technical: [
      {value: 'login_issue', label: 'Login Issue'},
      {value: 'bug_report', label: 'Bug Report'},
      {value: 'performance', label: 'Performance Problem'},
    ],
    account: [
      {value: 'profile_update', label: 'Profile Update'},
      {value: 'access_request', label: 'Access Request'},
      {value: 'subscription', label: 'Subscription Change'},
    ],
    enterprise: [
      {value: 'onboarding', label: 'Enterprise Onboarding'},
      {value: 'integration', label: 'Integration Support'},
      {value: 'contract', label: 'Contract Management'},
    ],
  };

  subcategory.innerHTML = '<option value="">Select subcategory...</option>';
  if(options[category]){
    options[category].forEach(item => {
      const opt = document.createElement('option');
      opt.value = item.value;
      opt.textContent = item.label;
      subcategory.appendChild(opt);
    });
  }
}

function renderTickets(tickets){
  const container = document.getElementById("ticket-list");
  if(tickets.length === 0){
    container.innerHTML = '<div class="empty-state"><i class="fas fa-inbox"></i><br>No tickets available</div>';
    return;
  }
  
  container.innerHTML = tickets.map(t=>`
    <div class="ticket-row">
      <div class="tic-id"><i class="fas fa-hashtag"></i>${t.id}</div>
      <div class="tic-content">
        <div class="tic-text">${t.text}</div>
        <div class="tic-meta">
          ${setPriority(t.priority)}
          <span><i class="fas fa-fire"></i> urgency=${t.urgency}</span>
          <span><i class="fas fa-clock"></i> sla=${t.sla_deadline}</span>
          <span><i class="fas fa-smile"></i> ${t.sentiment}</span>
          <span><i class="fas fa-user"></i> ${t.user_type}</span>
          ${t.category?`<span><i class="fas fa-tag"></i> ${t.category}</span>`:''}
          ${t.resolved?'<span class="resolved-badge"><i class="fas fa-check-circle"></i> RESOLVED</span>':''}
        </div>
      </div>
    </div>`).join("");
}

function addLog(action, reward, breakdown, custSatisfaction, teamEfficiency, longTermValue, info){
  const box=document.getElementById("log-box");
  if(box.querySelector('.empty-state'))box.innerHTML="";
  const el=document.createElement("div");
  el.className="entry";
  const good=reward>=0?"color:var(--green)":"color:var(--red)";
  const icons = {
    classify: 'tags',
    respond: 'comment',
    escalate: 'arrow-up',
    resolve: 'check',
    transfer: 'exchange-alt',
    schedule_callback: 'calendar-clock',
    request_supervisor: 'user-tie'
  };
  el.innerHTML=`<span class="tag ${action.action_type}"><i class="fas fa-${icons[action.action_type]||'cog'}"></i> ${action.action_type.toUpperCase()}</span>`+
    `<span style="font-weight:600">Ticket #${action.ticket_id}</span> `+
    `<span style="${good}; font-weight:600">${reward>=0?'+':''}${reward.toFixed(3)}</span> `+
    `<span style="color:var(--text-muted); margin-left:0.5rem">[Satisfaction: ${custSatisfaction.toFixed(2)}, Efficiency: ${teamEfficiency.toFixed(2)}, Value: ${longTermValue.toFixed(2)}]</span>`;
  box.prepend(el);
}

async function doReset(){
  cumReward=0;
  const obs=await api("/reset","POST");
  totalTickets=obs.tickets.length;
  document.getElementById("m-step").textContent=obs.step_count;
  document.getElementById("m-res").textContent=obs.tickets.filter(t=>t.resolved).length;
  document.getElementById("m-open").textContent=obs.tickets.filter(t=>!t.resolved).length;
  document.getElementById("m-reward").textContent="0.000";
  document.getElementById("progress-bar").style.width="0%";
  document.getElementById("progress-text").textContent="0%";
  document.getElementById("ticket-id").value=obs.current_ticket.id;
  document.getElementById("log-box").innerHTML='<div class="empty-state"><i class="fas fa-terminal"></i><br>Episode reset. Ready for training...</div>';
  renderTickets(obs.tickets);
}

async function doStep(){
  const actionType = document.getElementById("action-type").value;
  const action={
    action_type: actionType,
    ticket_id: parseInt(document.getElementById("ticket-id").value),
    category: document.getElementById("category")?.value || null,
    subcategory: document.getElementById("subcategory")?.value || null,
    response: document.getElementById("response-text")?.value || null,
    escalate_to: document.getElementById("escalate-to")?.value || null,
    transfer_to_agent: document.getElementById("transfer-to")?.value || null,
    callback_time: document.getElementById("callback-time")?.value || null,
  };
  
  try {
    const res=await api("/step","POST",action);
    cumReward+=res.reward;
    document.getElementById("m-step").textContent=res.observation.step_count;
    document.getElementById("m-res").textContent=res.observation.tickets.filter(t=>t.resolved).length;
    document.getElementById("m-open").textContent=res.observation.tickets.filter(t=>!t.resolved).length;
    document.getElementById("m-reward").textContent=res.reward.toFixed(3);
    const pct=Math.round((res.observation.tickets.filter(t=>t.resolved).length/totalTickets)*100);
    document.getElementById("progress-bar").style.width=pct+"%";
    document.getElementById("progress-text").textContent=pct+"%";
    addLog(action, res.reward, res.reward_breakdown, res.customer_satisfaction_impact, res.team_efficiency_impact, res.long_term_value_impact, res.info);
    renderTickets(res.observation.tickets);
    if(res.done){
      const resolved = res.info.resolved_count;
      const total = totalTickets;
      alert(`Episode Complete!\\n\\nPerformance Metrics:\\n- Resolved: ${resolved}/${total} tickets (${Math.round(resolved/total*100)}%)\\n- Total Steps: ${res.observation.step_count}\\n- Final Score: ${res.info.final_score || 'N/A'}`);
    }
  } catch(e) {
    alert("Error: " + e.message);
    console.error(e);
  }
}

// Auto-load state on page open
(async()=>{
  try{
    updateFormFields();
    const s=await api("/state");
    if(s.step_count>0){
      document.getElementById("m-step").textContent=s.step_count;
      document.getElementById("m-res").textContent=s.resolved_count;
      document.getElementById("m-open").textContent=s.open_count;
    }
  }catch(e){}
})();
</script>
</body>
</html>
"""


@app.get("/web", response_class=HTMLResponse, include_in_schema=False)
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def web_ui():
    """Interactive web UI for exploring the environment."""
    return WEB_UI


def main():
    """Main entry point for OpenEnv deployment."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()