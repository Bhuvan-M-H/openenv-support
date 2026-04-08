import random
from typing import Any, Dict, List, Set, Tuple
from datetime import datetime, time

from support_env.models import Action, Observation, Reward, Ticket, TeamMember, SystemStatus, CustomerProfile


class SupportEnv:
    """Advanced support-ticket simulator with multi-agent dynamics, customer profiles, and system complexity."""

    CATEGORY_RULES = {
        "billing": ("refund", "invoice", "charged", "payment", "subscription", "pricing"),
        "technical": ("crash", "login", "password", "error", "feature", "bug", "performance"),
        "account": ("cancel", "card", "details", "update", "access", "verification"),
        "enterprise": ("integration", "api", "deployment", "security", "compliance"),
    }

    SUBCATEGORY_RULES = {
        "billing": ["payment_methods", "refunds", "invoices", "subscriptions", "pricing"],
        "technical": ["login_issues", "performance", "bugs", "features", "compatibility"],
        "account": ["security", "verification", "updates", "cancellation", "access"],
        "enterprise": ["api_integration", "deployment", "security", "compliance", "training"],
    }

    def __init__(self, max_steps: int = 25, ticket_count: int = 8, seed: int = 42) -> None:
        self.max_steps = max_steps
        self.ticket_count = ticket_count
        self.step_count = 0
        self.history: List[str] = []
        self.tickets: List[Ticket] = []
        self.current_index = 0
        self.escalated_ids: Set[int] = set()
        self.random = random.Random(seed)

        # Advanced features
        self.team_members = self._initialize_team()
        self.system_status = self._initialize_system_status()
        self.customer_profiles: Dict[str, CustomerProfile] = {}
        self.pending_callbacks = 0
        self.business_hours_start = time(9, 0)  # 9 AM
        self.business_hours_end = time(17, 0)   # 5 PM
        self.reset()

    def _initialize_team(self) -> List[TeamMember]:
        """Initialize support team with different expertise levels."""
        team = [
            TeamMember(
                id="agent_1",
                name="Sarah Johnson",
                expertise=["billing", "account"],
                current_workload=0,
                availability_status="available",
                specialization_score={"billing": 0.9, "account": 0.8, "technical": 0.3}
            ),
            TeamMember(
                id="agent_2",
                name="Mike Chen",
                expertise=["technical", "enterprise"],
                current_workload=0,
                availability_status="available",
                specialization_score={"technical": 0.95, "enterprise": 0.85, "billing": 0.2}
            ),
            TeamMember(
                id="supervisor_1",
                name="Dr. Emily Rodriguez",
                expertise=["billing", "technical", "account", "enterprise"],
                current_workload=0,
                availability_status="available",
                specialization_score={"billing": 0.8, "technical": 0.8, "account": 0.8, "enterprise": 0.9}
            ),
        ]
        return team

    def _initialize_system_status(self) -> SystemStatus:
        """Initialize system status with realistic variations."""
        return SystemStatus(
            knowledge_base_health=self.random.uniform(0.7, 1.0),
            system_uptime=self.random.uniform(0.85, 0.99),
            peak_hours=self.random.choice([True, False]),
            active_incidents=[] if self.random.random() > 0.3 else ["API slowdown", "Email service delay"]
        )

    def _generate_customer_profile(self) -> CustomerProfile:
        """Generate realistic customer profiles."""
        customer_id = f"CUST_{self.random.randint(1000, 9999)}"
        lifetime_value = self.random.choice([0, self.random.uniform(50, 5000)])  # Free users = $0

        profile = CustomerProfile(
            customer_id=customer_id,
            lifetime_value=round(lifetime_value, 2),
            account_age_days=self.random.randint(1, 365*3),  # Up to 3 years
            previous_tickets_count=self.random.randint(0, 15),
            satisfaction_history=[round(self.random.uniform(0.3, 1.0), 2) for _ in range(min(5, self.random.randint(0, 5)))],
            preferred_contact_method=self.random.choice(["email", "chat", "phone"]),
            vip_status=lifetime_value > 1000 or self.random.random() < 0.1
        )

        self.customer_profiles[customer_id] = profile
        return profile

    def _make_tickets(self) -> List[Ticket]:
        """Generate diverse, realistic support tickets with customer profiles."""
        ticket_templates = [
            # Billing issues
            ("Refund not processed after duplicate payment charged", "billing", "refunds"),
            ("Invoice shows incorrect amount for enterprise subscription", "billing", "invoices"),
            ("Payment method declined despite valid card details", "billing", "payment_methods"),
            ("Subscription auto-renewed at wrong pricing tier", "billing", "subscriptions"),

            # Technical issues
            ("App crashes on login after latest update", "technical", "login_issues"),
            ("Performance degraded significantly after feature update", "technical", "performance"),
            ("New feature not working as described in documentation", "technical", "features"),
            ("Compatibility issues with latest OS version", "technical", "compatibility"),

            # Account issues
            ("Unable to update billing address and contact details", "account", "updates"),
            ("Two-factor authentication not working with backup codes", "account", "security"),
            ("Account verification email not received", "account", "verification"),
            ("Premium features inaccessible despite active subscription", "account", "access"),

            # Enterprise issues
            ("API integration failing with authentication errors", "enterprise", "api_integration"),
            ("Deployment pipeline stuck in provisioning stage", "enterprise", "deployment"),
            ("Security audit flagged compliance violations", "enterprise", "compliance"),
            ("Training materials not accessible for new team members", "enterprise", "training"),
        ]

        priorities = ["low", "medium", "high", "urgent"]
        sentiments = ["angry", "neutral", "happy", "frustrated", "delighted"]
        user_types = ["free", "premium", "enterprise"]

        tickets: List[Ticket] = []

        for idx in range(self.ticket_count):
            template, category, subcategory = self.random.choice(ticket_templates)
            priority = self.random.choice(priorities)
            user_type = self.random.choice(user_types)

            # Adjust urgency based on priority and user type
            urgency_floor = {"low": 1, "medium": 3, "high": 6, "urgent": 8}[priority]
            urgency_bonus = {"enterprise": 2, "premium": 1, "free": 0}[user_type]
            urgency = min(10, self.random.randint(urgency_floor, urgency_floor + 3) + urgency_bonus)

            # SLA deadline based on priority and user type
            base_sla = {"low": 20, "medium": 15, "high": 10, "urgent": 5}[priority]
            sla_bonus = {"enterprise": -3, "premium": -1, "free": 0}[user_type]
            sla_deadline = max(1, base_sla + sla_bonus + self.random.randint(-2, 2))

            # Generate customer profile
            customer_profile = self._generate_customer_profile()

            # Determine resolution complexity
            complexity_factors = [
                priority in ["high", "urgent"],
                user_type == "enterprise",
                customer_profile.previous_tickets_count > 5,
                urgency > 7
            ]
            complexity_score = sum(complexity_factors)
            complexity = ["simple", "moderate", "complex"][min(2, complexity_score)]

            # Generate dynamic tags
            tags = []
            if customer_profile.vip_status:
                tags.append("vip_customer")
            if customer_profile.previous_tickets_count > 3:
                tags.append("repeat_customer")
            if customer_profile.lifetime_value > 1000:
                tags.append("high_value")
            if priority == "urgent":
                tags.append("urgent")
            if user_type == "enterprise":
                tags.append("enterprise")

            # Adjust sentiment based on customer history
            base_sentiment = self.random.choice(sentiments)
            if customer_profile.satisfaction_history and sum(customer_profile.satisfaction_history) / len(customer_profile.satisfaction_history) < 0.6:
                # Dissatisfied customers more likely to be angry/frustrated
                base_sentiment = self.random.choice(["angry", "frustrated", "neutral"])

            tickets.append(
                Ticket(
                    id=idx + 1,
                    text=template,
                    user_type=user_type,
                    priority=priority,
                    sentiment=base_sentiment,
                    urgency=urgency,
                    sla_deadline=sla_deadline,
                    category=None,
                    subcategory=None,
                    customer_profile=customer_profile,
                    tags=tags,
                    resolution_complexity=complexity,
                )
            )

        return tickets

    def _next_open_ticket_index(self) -> int:
        open_indices = [i for i, ticket in enumerate(self.tickets) if not ticket.resolved]
        if not open_indices:
            return 0
        return min(open_indices, key=lambda i: (-self.tickets[i].urgency, self.tickets[i].sla_deadline))

    def _observation(self) -> Observation:
        return Observation(
            tickets=self.tickets,
            current_ticket=self.tickets[self.current_index],
            step_count=self.step_count,
            history=self.history,
        )

    def _expected_category(self, text: str) -> str:
        text_l = text.lower()
        for category, keywords in self.CATEGORY_RULES.items():
            if any(k in text_l for k in keywords):
                return category
        return "technical"

    @staticmethod
    def _clamp(score: float) -> float:
        return max(0.001, min(0.999, score))

    def _degrade_sla(self) -> int:
        breached = 0
        for ticket in self.tickets:
            if ticket.resolved:
                continue
            ticket.sla_deadline -= 1
            if ticket.sla_deadline <= 0:
                breached += 1
                ticket.urgency = min(10, ticket.urgency + 1)
                if ticket.sentiment == "neutral":
                    ticket.sentiment = "angry"
        return breached

    def _classify_score(self, ticket: Ticket, action: Action, breakdown: Dict[str, float]) -> float:
        if not action.category:
            breakdown["classification_missing"] = -0.099
            return -0.099
        expected = self._expected_category(ticket.text)
        if action.category == expected:
            breakdown["classification_accuracy"] = 0.35
            return 0.35
        breakdown["classification_accuracy"] = -0.099
        return -0.099

    def _response_score(self, ticket: Ticket, action: Action, breakdown: Dict[str, float]) -> float:
        if not action.response:
            breakdown["response_missing"] = -0.149
            return -0.149
        response = action.response.lower()
        length_bonus = 0.1 if len(response) > 30 else 0.03
        empathy_bonus = 0.1 if any(w in response for w in ("sorry", "understand", "prioritizing")) else 0.001
        urgency_bonus = 0.08 if ticket.urgency > 7 and "priorit" in response else 0.001
        total = 0.1 + length_bonus + empathy_bonus + urgency_bonus
        breakdown["response_quality"] = total
        return total

    def reset(self) -> Observation:
        self.step_count = 0
        self.history = []
        self.tickets = self._make_tickets()
        self.current_index = 0
        self.escalated_ids.clear()
        return self._observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        self.step_count += 1
        breakdown: Dict[str, float] = {}
        score = 0.001
        customer_satisfaction_impact = 0.001
        team_efficiency_impact = 0.001
        long_term_value_impact = 0.001
        info: Dict[str, Any] = {}

        ticket = next((t for t in self.tickets if t.id == action.ticket_id), None)
        if ticket is None:
            breakdown["invalid_ticket"] = -0.4
            score -= 0.4
            customer_satisfaction_impact -= 0.3
        elif ticket.resolved:
            breakdown["already_resolved"] = -0.2
            score -= 0.2
            customer_satisfaction_impact -= 0.1
        else:
            if action.action_type == "classify":
                score += self._classify_score(ticket, action, breakdown)
                if action.category:
                    ticket.category = action.category
                if action.subcategory:
                    ticket.subcategory = action.subcategory
                customer_satisfaction_impact += 0.1  # Organization shows understanding

            elif action.action_type == "respond":
                score += self._response_score(ticket, action, breakdown)
                if action.response:
                    ticket.response = action.response
                # Sentiment-based response quality
                if ticket.sentiment in ["angry", "frustrated"] and "sorry" in action.response.lower():
                    customer_satisfaction_impact += 0.2
                elif ticket.sentiment == "happy" and "thank" in action.response.lower():
                    customer_satisfaction_impact += 0.1

            elif action.action_type == "escalate":
                self.escalated_ids.add(ticket.id)
                ticket.escalated_to = action.escalate_to or "supervisor"
                escalate_bonus = 0.25 if ticket.urgency > 7 or ticket.priority in ["high", "urgent"] else 0.05
                breakdown["escalation_fit"] = escalate_bonus
                score += escalate_bonus
                team_efficiency_impact += 0.1  # Proper escalation improves team efficiency

            elif action.action_type == "transfer":
                # Transfer to specific agent
                if action.transfer_to_agent:
                    target_agent = next((a for a in self.team_members if a.id == action.transfer_to_agent), None)
                    if target_agent and target_agent.availability_status == "available":
                        target_agent.current_workload += 1
                        breakdown["successful_transfer"] = 0.15
                        score += 0.15
                        team_efficiency_impact += 0.2
                    else:
                        breakdown["transfer_failed"] = -0.1
                        score -= 0.1
                        customer_satisfaction_impact -= 0.1

            elif action.action_type == "search_kb":
                # Knowledge base search action
                kb_effectiveness = self.system_status.knowledge_base_health
                search_bonus = 0.1 * kb_effectiveness
                breakdown["kb_search"] = search_bonus
                score += search_bonus
                team_efficiency_impact += 0.05
                query_text = (action.search_query or ticket.text or "general support search").strip()
                query_lower = query_text.lower()
                profile_note = ""
                if ticket.user_type == "enterprise":
                    profile_note = " This is an enterprise customer issue."
                elif ticket.user_type == "premium":
                    profile_note = " This is a premium customer issue."

                title = "Knowledge Base Article"
                summary = f"A relevant knowledge base article was located for your query. Ticket title: '{ticket.text}'.{profile_note}"
                recommended = "Use the article guidance to resolve the ticket or escalate if needed."

                if "refund" in query_lower or "payment" in query_lower or "invoice" in query_lower:
                    title = "Disputed Payment & Refund Processing"
                    summary = (
                        f"This article explains how to verify payment authorization, identify duplicate charges, and process refunds for incorrect billing. "
                        f"Ticket issue: '{ticket.text}'.{profile_note}"
                    )
                    recommended = "Verify the invoice, confirm the payment source, and issue a refund or credit if the charge was incorrect."
                elif "login" in query_lower or "authentication" in query_lower or "two-factor" in query_lower or "2fa" in query_lower:
                    title = "Login Failure & 2FA Troubleshooting"
                    summary = (
                        f"This article walks through account recovery, authentication resets, and two-factor login issues. "
                        f"Ticket issue: '{ticket.text}'.{profile_note}"
                    )
                    recommended = "Confirm the customer's credentials, check 2FA device status, and reset sessions if necessary."
                elif "api" in query_lower or "integration" in query_lower or "deployment" in query_lower:
                    title = "API Integration Error Resolution"
                    summary = (
                        f"This article details API authentication failures, integration token problems, and enterprise deployment troubleshooting. "
                        f"Ticket issue: '{ticket.text}'.{profile_note}"
                    )
                    recommended = "Review API credentials, verify endpoint access, and ensure the integration settings match the customer's environment."
                elif "subscription" in query_lower or "pricing" in query_lower or "renewal" in query_lower:
                    title = "Subscription Billing and Renewal Support"
                    summary = (
                        f"This article explains subscription tiers, auto-renewal behavior, and correcting plan charge errors. "
                        f"Ticket issue: '{ticket.text}'.{profile_note}"
                    )
                    recommended = "Confirm the active plan, correct any billing tier mismatch, and update renewal settings as needed."
                elif "training" in query_lower or "access" in query_lower or "onboarding" in query_lower:
                    title = "Training Access and Onboarding Troubleshooting"
                    summary = (
                        f"This article covers restoring access to training materials, managing permissions, and resolving onboarding issues. "
                        f"Ticket issue: '{ticket.text}'.{profile_note}"
                    )
                    recommended = "Check account permissions, restore portal access, and verify the customer's training environment."
                elif "compat" in query_lower or "compatibility" in query_lower or "os version" in query_lower:
                    title = "Compatibility with Latest OS Versions"
                    summary = (
                        f"This article helps diagnose operating system compatibility issues and recommends updates or patches. "
                        f"Ticket issue: '{ticket.text}'.{profile_note}"
                    )
                    recommended = "Verify the customer's OS version, apply the latest compatibility patch, and confirm supported hardware requirements."
                elif "security" in query_lower or "compliance" in query_lower or "verification" in query_lower:
                    title = "Security and Compliance Troubleshooting"
                    summary = (
                        f"This article outlines how to resolve security verification problems, compliance flags, and account access issues. "
                        f"Ticket issue: '{ticket.text}'.{profile_note}"
                    )
                    recommended = "Review security logs, verify identity factors, and ensure compliance checks are completed successfully."
                else:
                    title = "Relevant Knowledge Base Guidance"
                    summary = (
                        f"This article provides a focused troubleshooting path based on the ticket description. "
                        f"Ticket issue: '{ticket.text}'.{profile_note}"
                    )
                    recommended = "Follow the article steps and escalate to a specialist if deeper investigation is needed."

                info["kb_result"] = {
                    "title": title,
                    "summary": summary,
                    "recommended": recommended,
                    "query": query_text,
                }

            elif action.action_type == "schedule_callback":
                # Schedule callback for complex issues
                if ticket.resolution_complexity in ["moderate", "complex"]:
                    self.pending_callbacks += 1
                    breakdown["callback_scheduled"] = 0.1
                    score += 0.1
                    customer_satisfaction_impact += 0.15  # Shows commitment to resolution
                else:
                    breakdown["unnecessary_callback"] = -0.05
                    score -= 0.05

            elif action.action_type == "request_supervisor":
                # Request supervisor assistance
                supervisor = next((a for a in self.team_members if "supervisor" in a.id), None)
                if supervisor and supervisor.availability_status == "available":
                    supervisor.current_workload += 1
                    breakdown["supervisor_help"] = 0.2
                    score += 0.2
                    team_efficiency_impact += 0.15
                else:
                    breakdown["supervisor_unavailable"] = -0.1
                    score -= 0.1

            elif action.action_type == "resolve":
                has_context = bool(ticket.category and ticket.response)
                resolve_power = 0.95 if ticket.id in self.escalated_ids else 0.75

                # Complexity affects resolution success
                complexity_penalty = {"simple": 0, "moderate": 0.1, "complex": 0.2}[ticket.resolution_complexity or "simple"]
                resolve_penalty = ticket.urgency * 0.015 + complexity_penalty

                success_chance = max(0.1, resolve_power - resolve_penalty)
                if not has_context:
                    success_chance -= 0.2

                if self.random.random() < success_chance:
                    ticket.resolved = True
                    resolve_reward = 0.7 + (0.2 if ticket.sla_deadline > 0 else 0.001)

                    # Customer satisfaction bonus for good resolution
                    if ticket.customer_profile and ticket.customer_profile.satisfaction_history:
                        avg_satisfaction = sum(ticket.customer_profile.satisfaction_history) / len(ticket.customer_profile.satisfaction_history)
                        satisfaction_bonus = 0.1 if avg_satisfaction > 0.7 else 0.001
                        resolve_reward += satisfaction_bonus
                        customer_satisfaction_impact += satisfaction_bonus

                    # Long-term value impact for VIP customers
                    if ticket.customer_profile and ticket.customer_profile.vip_status:
                        long_term_value_impact += 0.1

                    breakdown["resolve_success"] = resolve_reward
                    score += resolve_reward
                else:
                    fail_penalty = -0.2 if has_context else -0.35
                    breakdown["resolve_failure"] = fail_penalty
                    score += fail_penalty
                    customer_satisfaction_impact -= 0.2

        # Update team workloads and availability
        self._update_team_status()

        # SLA degradation with business hours consideration
        breached = self._degrade_sla()
        if breached:
            sla_penalty = -0.08 * breached
            breakdown["sla_breach_penalty"] = sla_penalty
            score += sla_penalty
            customer_satisfaction_impact -= 0.1 * breached

        # Enhanced churn risk calculation
        churn_risk = sum(
            1 for t in self.tickets
            if not t.resolved and
            t.sentiment in ["angry", "frustrated"] and
            t.customer_profile and
            (t.customer_profile.vip_status or t.customer_profile.lifetime_value > 500) and
            t.sla_deadline <= 0
        )
        if churn_risk:
            churn_penalty = -0.04 * churn_risk
            breakdown["premium_churn_risk"] = churn_penalty
            score += churn_penalty
            long_term_value_impact -= 0.05 * churn_risk

        # System status impact
        if self.system_status.active_incidents:
            system_penalty = -0.02 * len(self.system_status.active_incidents)
            breakdown["system_issues"] = system_penalty
            score += system_penalty
            team_efficiency_impact -= 0.05

        score = self._clamp(score)

        # Enhanced history logging
        self.history.append(
            f"step={self.step_count} action={action.action_type} ticket={action.ticket_id} "
            f"score={score:.3f} satisfaction={customer_satisfaction_impact:.3f} "
            f"efficiency={team_efficiency_impact:.3f} value={long_term_value_impact:.3f} "
            f"details={breakdown}"
        )

        self.current_index = self._next_open_ticket_index()
        done = self.step_count >= self.max_steps or all(t.resolved for t in self.tickets)

        if not isinstance(info, dict):
            info = {}
        info.update({
            "breached_tickets": breached,
            "resolved_count": sum(t.resolved for t in self.tickets),
            "open_count": sum(not t.resolved for t in self.tickets),
            "escalated_open": sum(1 for t in self.tickets if t.id in self.escalated_ids and not t.resolved),
            "pending_callbacks": self.pending_callbacks,
            "team_utilization": sum(a.current_workload for a in self.team_members) / len(self.team_members),
            "customer_satisfaction_impact": customer_satisfaction_impact,
            "team_efficiency_impact": team_efficiency_impact,
            "long_term_value_impact": long_term_value_impact,
        })

        # Enhanced reward breakdown
        enhanced_breakdown = breakdown.copy()
        enhanced_breakdown.update({
            "customer_satisfaction_impact": customer_satisfaction_impact,
            "team_efficiency_impact": team_efficiency_impact,
            "long_term_value_impact": long_term_value_impact,
        })

        return self._observation(), Reward(
            score=score,
            breakdown=enhanced_breakdown,
            customer_satisfaction_impact=customer_satisfaction_impact,
            team_efficiency_impact=team_efficiency_impact,
            long_term_value_impact=long_term_value_impact
        ), done, info

    def _update_team_status(self):
        """Update team member availability based on workload."""
        for agent in self.team_members:
            if agent.current_workload > 3:
                agent.availability_status = "busy"
            elif agent.current_workload > 5:
                agent.availability_status = "offline"
            else:
                agent.availability_status = "available"

    def _is_business_hours(self) -> bool:
        """Check if current time is within business hours."""
        now = datetime.now().time()
        return self.business_hours_start <= now <= self.business_hours_end

    def _observation(self) -> Observation:
        """Create enhanced observation with team and system status."""
        if not self.tickets:
            raise ValueError("No tickets available in the environment.")
        if self.current_index < 0 or self.current_index >= len(self.tickets):
            self.current_index = 0
        return Observation(
            tickets=self.tickets,
            current_ticket=self.tickets[self.current_index],
            step_count=self.step_count,
            history=self.history,
            team_members=self.team_members,
            system_status=self.system_status,
            business_hours=self._is_business_hours(),
            pending_callbacks=self.pending_callbacks,
        )
