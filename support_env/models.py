from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict


class CustomerProfile(BaseModel):
    customer_id: str
    lifetime_value: float  # $ spent
    account_age_days: int
    previous_tickets_count: int
    satisfaction_history: List[float]  # Previous satisfaction scores
    preferred_contact_method: Literal["email", "chat", "phone"]
    vip_status: bool


class Ticket(BaseModel):
    id: int
    text: str
    user_type: Literal["free", "premium", "enterprise"]
    priority: Literal["low", "medium", "high", "urgent"]
    sentiment: Literal["angry", "neutral", "happy", "frustrated", "delighted"]
    urgency: int  # 1-10 scale
    sla_deadline: int  # steps remaining
    category: Optional[str] = None
    subcategory: Optional[str] = None
    resolved: bool = False
    response: Optional[str] = None
    escalated_to: Optional[str] = None
    customer_profile: Optional[CustomerProfile] = None
    tags: List[str] = []  # Dynamic tags like "repeat_customer", "high_value"
    resolution_complexity: Optional[Literal["simple", "moderate", "complex"]] = None


class TeamMember(BaseModel):
    id: str
    name: str
    expertise: List[str]  # ["billing", "technical", "account"]
    current_workload: int  # Active tickets
    availability_status: Literal["available", "busy", "offline"]
    specialization_score: Dict[str, float]  # Category -> expertise level


class SystemStatus(BaseModel):
    knowledge_base_health: float  # 0.0-1.0
    system_uptime: float  # 0.0-1.0
    peak_hours: bool
    active_incidents: List[str]  # Current system issues


class Observation(BaseModel):
    tickets: List[Ticket]
    current_ticket: Ticket
    step_count: int
    history: List[str]
    team_members: List[TeamMember]
    system_status: SystemStatus
    business_hours: bool
    pending_callbacks: int  # Customers waiting for callbacks


class Action(BaseModel):
    action_type: Literal["classify", "respond", "resolve", "escalate", "transfer", "search_kb", "schedule_callback", "request_supervisor"]
    ticket_id: int
    category: Optional[str] = None
    subcategory: Optional[str] = None
    response: Optional[str] = None
    escalate_to: Optional[str] = None  # Specific team/department
    transfer_to_agent: Optional[str] = None
    search_query: Optional[str] = None
    callback_time: Optional[str] = None


class Reward(BaseModel):
    score: float = Field(gt=0.0, lt=1.0)
    breakdown: Dict[str, float]
    customer_satisfaction_impact: float
    team_efficiency_impact: float
    long_term_value_impact: float
