# Conference Talk Submission Package
## "Deterministic Context-Aware AI: Building Enterprise Copilots That Actually Work"

**Alternative Title:** "Beyond the Blank Chat Box: How Report-Driven Context Makes AI Trustworthy"

**Speaker:** Pete Tamisin  
**Target Audience:** AI/ML Engineers, Solution Architects, Platform Teams, Tech Leaders  
**Talk Length:** 30-40 minutes (adaptable to 20-25 min for shorter slots)  
**Level:** Intermediate to Advanced

---

## SUBMISSION MATERIALS

### 1. SHORT ABSTRACT (100-150 words)
**Use for:** Twitter/social media, conference websites with character limits

Most AI copilots are context-aware but unpredictable: they parse conversation history, guess at user intent, and hope for the best. Enterprise operations need something better.

This talk introduces "deterministic context-aware AI"—an architecture where context is explicitly anchored to reports and entity selections rather than implicitly inferred from chat. When you select a job in the cost report, the system locks context to that entity's graph data and generates all subsequent prompts deterministically from that structure.

Using a production RAG system built for data platform operations, we'll demonstrate how entity-anchored context combined with GraphRAG and multi-corpus retrieval delivers AI that is explainable, debuggable, and trustworthy. You'll learn how to build context systems where every prompt is a deterministic function of report state and drill-down selections.

---

### 2. LONG ABSTRACT (250-400 words)
**Use for:** Primary CFP submission, detailed conference requirements

**The Problem with Conversation-Based Context**

Every company is building AI copilots. Most are "context-aware"—they maintain conversation history and try to understand what users mean across multiple messages. But for enterprise operations, this approach fails catastrophically.

The fundamental issue: context inferred from conversation is fragile, ambiguous, and non-deterministic. When a user asks "Why is this job expensive?" after looking at a cost report, the AI must guess which job, retrieve its data, and hope it understood correctly. The result? Hallucinations, inconsistent answers, and operators who don't trust the system.

**The Solution: Deterministic Context-Aware AI**

This talk introduces an architecture pattern that makes context-awareness deterministic and reliable. Instead of inferring context from chat:

- **Reports establish context** (cost analysis, reliability metrics, performance data)
- **Selections anchor context to entities** (click a job → context locks to job_123's graph data)
- **Context persists as structured data** (graph + metadata, not conversation text)
- **All prompts are deterministic functions** of report state + entity selection
- **LLMs explain results** grounded in that locked context

The result is an AI copilot that maintains perfect context across complex workflows while being completely explainable and debuggable.

**What You'll Learn**

Through a live demonstration of a production system analyzing Databricks usage data (cost, reliability, performance), attendees will see:

1. **How entity-anchored context works**: Reports → Selections → Graph Context → Deterministic Prompts
2. **Why this is superior to chat-based context**: Comparison of conversation parsing vs explicit entity locking
3. **Building context-aware action systems**: How drill-downs generate deterministic prompt templates
4. **Grounded recommendations without hallucinations**: Multi-corpus RAG (telemetry + docs + runbooks) with hard constraints on LLM outputs
5. **GraphRAG for operational reasoning**: Using knowledge graphs to maintain rich, structured context
6. **Production patterns**: Debug modes, context inspection, runbook versioning, citation enforcement

**Why This Matters**

Enterprises are investing billions in AI copilots, but most fail because they treat context as a conversation-parsing problem and let LLMs hallucinate recommendations. This talk provides a proven alternative that combines the power of context-awareness with the reliability and accountability enterprises require.

**Takeaways**: Architecture diagrams, open-source code reference, decision frameworks for building deterministic context systems, runbook templates.

---

### 3. ELEVATOR PITCH (30-50 words)
**Use for:** Quick pitches to organizers, social media teasers

Most AI copilots infer context from conversation and guess at intent. Learn how deterministic context-aware AI—where reports anchor context to entities and all prompts are generated from that structure—delivers trustworthy AI for enterprise operations.

---

### 4. KEY TAKEAWAYS (3-5 bullets)
**Use for:** Conference programs, attendee guides

Attendees will learn:

- How deterministic context-aware AI differs from conversation-based context: explicit entity anchoring vs implicit chat parsing
- Architecture patterns for building context systems where reports establish context and selections lock it to structured graph data
- How to generate all prompts deterministically from report state and drill-down selections, eliminating ambiguity
- Preventing hallucinations in recommendations: multi-corpus RAG with docs, runbooks, and telemetry constrained to "only use provided sources"
- Production lessons from building an enterprise AI copilot with persistent, debuggable context and grounded recommendations for cost/reliability analysis

---

### 5. TARGET AUDIENCE DESCRIPTION
**Use for:** CFP questions about who should attend

**Primary Audience:**
- AI/ML Engineers building enterprise copilots or AI assistants
- Solution Architects designing RAG systems for production
- Platform Engineers responsible for operational tooling
- Engineering Managers evaluating AI strategies

**Secondary Audience:**
- Data Engineers working with observability and FinOps tools
- Product Managers defining AI product requirements
- Technical Leaders making build-vs-buy decisions for AI

**Prerequisites:** 
- Basic understanding of LLMs and prompt engineering
- Familiarity with RAG (Retrieval-Augmented Generation) concepts
- Experience with production systems (any domain)

No prior knowledge of Databricks required—the patterns apply universally.

---

### 6. TALK OUTLINE (30-40 minute version)

#### **PART 1: The Problem with Conversation-Based Context (5 minutes)**

**Opening Hook** (1 min)
- Demo: Show a chat-first copilot with conversation-based context failing on an operational question
- User clicks job in a dashboard, then asks "Why is this so expensive?"
- Watch AI retrieve wrong job, mix up context, provide answer about different entity

**Why Context-Inference Fails** (2 min)
- The fragility of conversation parsing: "this job", "the expensive one", "that cluster"
- Context degrades over long conversations
- Ambiguity multiplies with drill-downs: "show me the failures" → "why did the first one happen?"
- No way to inspect what context the AI is actually using

**The Enterprise Cost** (2 min)
- Real examples: operator asks about job_123, AI answers about job_456
- Context drift leads to hallucinated recommendations
- Inconsistent answers across identical questions (context parsed differently)
- Engineers stop trusting the AI and go back to manual queries

**Transition:** "The problem isn't context-awareness itself—it's how we establish and maintain context. Let me show you a better way."

---

#### **PART 2: The Solution—Deterministic Context-Aware AI (7 minutes)**

**Core Principle** (2 min)
- "Context should be anchored to entities, not inferred from conversation"
- Reports establish what context is available
- Selections explicitly lock context to specific entities
- All prompts are deterministic functions of that locked context
- Context persists as structured graph data, not conversation text

**Architecture Overview** (3 min)
- Show architecture diagram:
  ```
  Report (establishes available context) → 
  User Selection (anchors to entity) → 
  Graph Context Loaded (job → cluster → events) →
  Context Persists in Session → 
  All Prompts Generated from This Context →
  LLM Explains Results
  ```
- Each layer's responsibility
- How context flows through the system
- Where LLMs fit (explanation layer working with locked context)

**Entity-Anchored Context in Action** (2 min)
- Show the difference:
  - **Chat-based**: "Show expensive jobs" → "Why is the first one costly?" (AI must parse "first one")
  - **Entity-anchored**: Click job_123 in report → context locks to job_123's graph → "Why is this expensive?" (unambiguous)
- Context includes: entity metadata, graph relationships, historical patterns
- Context remains locked until explicitly cleared or changed
- Every subsequent prompt operates on this same structured context

---

#### **PART 3: Live Demo—Entity-Anchored Context in Action (10 minutes)**

**Scenario Setup** (1 min)
- "You're a platform engineer. Jobs are costing more than expected."
- Pull up the Usage Copilot interface

**Demo Flow:**

1. **Report Establishes Available Context** (2 min)
   - Show "Job Cost" report (bar chart from SQL query)
   - This report defines what entities you can anchor context to
   - Each bar represents a potential context anchor

2. **Selection Anchors Context** (3 min)
   - Click on high-cost job → context anchor panel appears
   - Show what happened behind the scenes:
     - job_id locked in session state
     - Graph context loaded: job → cluster → usage → events
     - Action chips generated based on THIS entity's context
   - Highlight: context is now LOCKED to job_123

3. **Deterministic Prompts from Anchored Context** (2 min)
   - Click "Why is this job expensive?"
   - Show debug panel: exact prompt includes job_123's graph context
   - Response is grounded in the locked context—can't answer about wrong job
   - Citations link back to entities in the graph

4. **Context Persists Across Prompts** (2 min)
   - Click "How do I optimize this?"
   - Show: SAME context still locked (no re-inference needed)
   - Prompt template references same job_123 graph
   - **Multi-corpus retrieval happens here:**
     - Telemetry corpus: job_123's actual usage patterns
     - Documentation corpus: Databricks optimization best practices
     - Runbook corpus: Known optimization procedures for this job type
   - Show retrieved documents in debug panel
   - LLM constrained to: "Only use provided information and context"
   - Result: Grounded recommendations with citations to source docs

5. **Context Inspection & Clearing** (1 min)
   - Show context inspection panel: see the actual graph data
   - Show retrieved recommendation documents
   - Demonstrate clearing context (deselect entity)
   - Select different job → context re-anchors to new entity

**Key Point:** "Context is explicitly set, structurally maintained, and deterministically used. No guessing. No drift. No ambiguity."

---

#### **PART 4: Design Patterns for Deterministic Context (8 minutes)**

**Pattern 1: When to Use Entity-Anchored Context** (2 min)
- ✅ Operational workflows (debugging, cost analysis, incident response)
- ✅ Multi-step investigations (drill down → analyze → optimize)
- ✅ When context needs to persist across many questions
- ✅ Domain tools requiring precise entity references
- ❌ General exploration where context should flow naturally
- ❌ Creative/brainstorming sessions
- Comparison table: conversation-based vs entity-anchored

**Pattern 2: Building Context-Aware Report Systems** (2 min)
- How to design reports that establish context domains
- Mapping user drill-downs to entity anchoring
- Context hierarchies: workspace → job → run → task
- Progressive disclosure: start with overview, drill to detail
- Each level = potential context anchor

**Pattern 3: Graph-Based Context Structures** (2 min)
- Why graphs are ideal for maintaining context
- Entity extraction from operational data
- Relationship traversal provides rich context automatically
- Context = subgraph around anchor entity
- Combining graph context with vector retrieval

**Pattern 4: Grounded Recommendations (No Hallucinations)** (2 min)
- When users ask "How do I optimize this?", don't let LLM guess
- Create recommendation corpus: best practices + platform docs + runbooks
- Retrieve relevant recommendations using vector search
- Instruct LLM: "Only answer using provided information and context"
- Example: Runbooks in markdown → embed → index → retrieve → cite
- Result: Every recommendation is traceable to a source document

**Pattern 5: Context State Management** (1 min)
- Session-level context persistence
- When to clear vs maintain context
- Multi-entity context (comparing two jobs)
- Context inspection for debugging

---

#### **PART 5: Production Lessons (5 minutes)**

**What Worked** (2 min)
- Operators trust it because context is visible and inspectable
- Entity-anchored context eliminates "did it understand me?" anxiety
- Deterministic prompts from stable context are testable (regression suites)
- Debug mode showing exact context state turns skeptics into advocates
- Graph-based context naturally supports multi-hop reasoning

**What Was Hard** (2 min)
- Designing the corpus routing (which corpus should serve as the RAG source based on request)
- Context clearing UX: when to auto-clear vs persist
- Graph context can get large—need strategies for context window limits

**When Conversation-Based Context IS Better** (1 min)
- Open-ended exploration without clear entities ("tell me about pooling in general...")
- Learning and education (context should flow naturally)
- Quick one-off questions where anchoring is overhead
- **The key**: Know when you need deterministic context vs fluid context but still limit the source to approved corpuses to limit hallucinations.

---

#### **PART 6: Takeaways & Decision Framework (3-5 minutes)**

**Decision Framework** (2 min)
- Flowchart: "How Should I Handle Context in My AI System?"
  - **Need precise entity references?** → Entity-anchored context
  - **Multi-step workflows on same entity?** → Entity-anchored context
  - **High stakes if wrong entity referenced?** → Entity-anchored context
  - **Open exploration, fluid conversation?** → Conversation-based context
  - **Hybrid needs?** → Start with entity-anchored, add chat for edge cases

**Architecture Reference** (1 min)
- Share GitHub repo (Usage Copilot)
- Architecture diagrams: context flow, graph structure, prompt generation
- Open for learning (proprietary data layer removed)
- Code shows: context anchoring, graph loading, prompt templates

**Call to Action** (1 min)
- "Rethink how your AI maintains context"
- "Don't default to conversation parsing for everything"
- "Build AI where context is a first-class architectural concern"
- "Make context explicit, structural, and inspectable"

**Q&A** (remaining time)

---

### 7. SLIDES OUTLINE (35-40 slides)

**Section 1: The Problem (Slides 1-10)**
1. Title slide: "Deterministic Context-Aware AI"
2. "Every company is building context-aware AI copilots"
3. "Most use conversation-based context" [screenshot: chat interface]
4. Demo video: Context inference failing (asks about wrong entity)
5. "Why conversation-based context fails" [diagram: context drift over time]
6. "The ambiguity problem" [examples: "this job", "the expensive one", "that cluster"]
7. "The enterprise cost" [3 scenarios: wrong entity, context drift, trust erosion]
8. "Engineers lose trust when context is unreliable"
9. "Context IS important—we just need to make it deterministic"
10. Section break

**Section 2: Solution—Deterministic Context (Slides 11-20)**
11. Core principle: "Anchor context to entities, not conversations"
12. "What is entity-anchored context?" [diagram: report → selection → locked context]
13. Architecture diagram (overview): Context flow from reports to prompts
14. Architecture diagram (detailed): Graph context loading and persistence
15. "Conversation-based vs Entity-anchored" [comparison table]
16. "Context as structured graph data" [graph visualization]
17. "All prompts generated from locked context" [code snippet showing template + context]
18. Example: Context lifecycle (select → anchor → persist → use → clear)
19. "Context is inspectable and debuggable" [debug panel screenshot]
20. Section break

**Section 3: Demo—Context in Action (Slides 21-28)**
21. "Let's see entity-anchored context in action"
22. Scenario setup: Platform engineer investigating costs
23. Screenshot: Report view (available context entities)
24. Screenshot: Click job → context anchors to job_123
25. Screenshot: Graph context loaded (job → cluster → events)
26. Screenshot: First prompt uses locked context (with debug view)
27. Screenshot: Second prompt STILL uses same context (persistence)
28. Demo summary: "Context is explicit. Persistent. Debuggable."

**Section 4: Context Architecture Patterns (Slides 29-36)**
29. "Design patterns for deterministic context"
30. Pattern 1: When to use entity-anchored context [decision flowchart]
31. Pattern 2: Building report-based context systems [report taxonomy]
32. Pattern 3: Graph-based context structures [graph schema + traversal]
33. Pattern 4: Grounded recommendations (no hallucinations) [multi-corpus diagram]
34. Example: Runbook ingestion pipeline [markdown → embed → retrieve → cite]
35. "Code is open source" [GitHub QR code + architecture diagrams]
36. Section break

**Section 5: Production Lessons (Slides 36-40)**
36. "What worked: Entity-anchored context builds trust"
37. "What was hard: Teaching 'select first, ask second' UX"
38. "When conversation-based context IS better"
39. Decision framework: "How should I handle context?" [flowchart]
40. "Make context a first-class architectural concern"

**Closing (Slides 41-43)**
41. Key takeaways (3 bullets)
42. Resources & GitHub link
43. Thank you + contact info

---

### 8. DEMO SCRIPT (For Practice & Backup)

**Pre-Demo Setup:**
- Have Usage Copilot running locally (in case of WiFi issues)
- Screen recording as backup
- Two browser tabs: live demo + backup video
- Test all click interactions before going on stage

**Demo Narration (verbatim):**

*[Pull up Usage Copilot interface]*

"Imagine you're a platform engineer and your manager just asked: 'Why did our Databricks costs spike last week?' With conversation-based context, you'd type that question and hope the AI understands which jobs you mean.

*[Show chat interface attempting same question]*

See how it's answering about the wrong job? It inferred context from the conversation, but got it wrong. That's the fragility of conversation-based context.

*[Switch to Usage Copilot]*

With deterministic context-aware AI, we start differently. Here's a report showing job costs—this establishes what entities we can anchor context to.

*[Click on high-cost job in bar chart]*

When I select this job, watch what happens. The system explicitly anchors context to job_123. Behind the scenes, it's loading this job's graph context: the job, its cluster, usage patterns, and any failures.

*[Show context panel]*

This is the locked context. It's not inferred from conversation—it's explicitly tied to the entity I selected. And it persists across all my questions about this job.

*[Hover over action options]*

Now the system shows me actions I can take, all grounded in THIS specific job's context.

*[Click 'Why is this job expensive?']*

When I ask a question, the system generates a prompt that includes the locked context. Let me show you.

*[Toggle debug panel]*

See? The prompt includes job_123's ID, its cost data, its cluster configuration—all from the locked graph context. The LLM can't answer about the wrong job because the context is structurally anchored.

*[Show response with citations]*

And the answer includes citations linking back to the graph data. Full transparency.

*[Click 'How do I optimize this?']*

Watch this—I'm asking for recommendations now. Here's where the system prevents hallucinations.

*[Show debug panel - retrieval phase]*

The system retrieves from three corpora:
- Telemetry: job_123's actual usage patterns
- Documentation: Databricks optimization best practices  
- Runbooks: Pre-written optimization procedures for this job type

*[Show retrieved documents panel]*

See these documents? They're the ONLY information the LLM is allowed to use. The prompt explicitly says: 'Only answer using the provided information and context. Do not speculate or add recommendations not found in the sources.'

*[Show response with multiple citations]*

Every recommendation cites a source document. If we don't have a runbook for this scenario, the system says so—it doesn't make something up.

This is how you get trustworthy recommendations: retrieve from a curated corpus, constrain the LLM to only use that, and cite everything.

*[Show context persistence indicator]*

And the context is STILL locked to job_123. No re-inference. Same deterministic context.

*[Click context inspection view]*

I can even inspect the exact graph context the AI is using. This is what makes it debuggable and trustworthy.

That's deterministic context-aware AI. Context is explicitly anchored to entities, maintained as structured graph data, and every prompt is a deterministic function of that context. No guessing. No drift. Just reliable AI for production systems."

---

### 9. SPEAKER BIO (Multiple Versions)

**Short Bio (50 words)**

Pete Tamisin is Director of Solution Architecture at Capital One Software and former Tech Lead at Databricks. With 20+ years architecting data and AI platforms for Fortune 500 companies, he's passionate about building trustworthy AI systems. He recently built a deterministic RAG copilot combining GraphRAG with report-driven prompt engineering.

**Medium Bio (100 words)**

Pete Tamisin is Director of Solution Architecture at Capital One Software (via acquisition of Sync Computing), where he leads technical enablement for enterprise data platforms. Previously at Databricks and multiple startups, Pete has spent over 20 years designing data engineering and AI systems for Fortune 500 companies including Siemens, Motorola, and ABB. He's focused on building AI systems that operators actually trust, recently architecting a deterministic RAG copilot that combines knowledge graphs with multi-corpus retrieval for cost optimization and reliability analysis. Pete is passionate about teaching and helping teams ship real-world AI systems at scale.

**Long Bio (150 words)**

Pete Tamisin is Director of Solution Architecture at Capital One Software (via acquisition of Sync Computing), where he leads the technical sales enablement and post-sales adoption strategy for enterprise data platforms. Previously a Tech Lead in Customer Success at Databricks, Pete managed portfolios exceeding $8M ARR and drove 200% business growth through strategic expansion initiatives. 

With over 20 years architecting data engineering and AI platforms for Fortune 500 companies—including Siemens, Motorola, and ABB—Pete brings deep expertise in Apache Spark, modern data lakehouses, and production AI systems. He's a Databricks Certified Generative AI Engineer and former Impact Player of the Year.

Pete is passionate about building AI systems that earn operator trust. His recent work includes architecting deterministic RAG copilots that combine GraphRAG with report-driven prompt engineering for operational intelligence. He regularly speaks about data engineering, AI architecture, and the intersection of trust and automation in enterprise systems.

---

### 10. SOCIAL MEDIA COPY (Ready to Post)

**LinkedIn Announcement Post:**

🎯 Excited to announce I'm submitting talks to AI/ML conferences on a topic that's critical for enterprise AI: "Deterministic Context-Aware AI"

Most AI copilots are context-aware—they maintain conversation history and try to understand references across messages. But for enterprise operations, this approach is too fragile.

The problem: Context inferred from conversation is ambiguous and non-deterministic. When operators drill down through reports and ask "Why is this expensive?", the AI must guess which entity they mean. Get it wrong, and trust evaporates.

There's a better way: **entity-anchored context**, where:
- Reports establish what entities are available
- User selections explicitly lock context to specific entities (job_123, not "that expensive job")
- Context persists as structured graph data, not conversation text
- All prompts are deterministic functions of that locked context

I've built this architecture in production for data platform operations (cost analysis, reliability debugging, performance optimization). Operators trust it because they can see exactly what context the AI is using.

In the talk, I'll demo the system live, show how context flows from reports → selections → graph loading → deterministic prompts, and provide architectural patterns for building context systems that earn trust.

If you're building enterprise AI where context precision matters, this is for you.

🔗 GitHub: [link]  
📧 Interested in having me speak at your event? DM me.

#AI #MachineLearning #ContextAwareAI #EnterpriseAI #RAG

---

**Twitter Thread:**

🧵 I'm submitting talks on deterministic context-aware AI for enterprise operations. Here's why this matters:

1/ Most AI copilots are "context-aware"—they maintain conversation history and try to understand what you mean across multiple messages.

But for enterprise ops, conversation-based context is too fragile and ambiguous.

2/ The problem: When you drill through cost reports and ask "Why is this so expensive?", the AI must infer which entity you mean from conversation.

"This job"? "The expensive one"? "That cluster"?

Get it wrong → wrong answer → lost trust.

3/ The solution: Entity-anchored context

Instead of inferring context from chat:
- Reports establish available entities
- Selections explicitly lock context to specific entities
- Context persists as structured graph data
- All prompts generated deterministically from locked context

4/ I built this for data platform ops. The architecture:

Report → Selection (anchors to job_123) → Graph Context Loaded → Context Persists → All Prompts Use Same Context → LLM Explains

Every question about that job uses identical, locked context. Zero ambiguity.

5/ Key difference from conversation-based:

Chat-based: "Show expensive jobs" → "Why is the first one costly?" (AI must parse "first one")

Entity-anchored: Click job_123 → context locks → "Why is this expensive?" (unambiguous—locked to job_123)

6/ In my talk, I'll:
- Demo live system showing context anchoring
- Show how context flows: report → selection → graph → prompts
- Provide patterns for building deterministic context systems
- Compare conversation-based vs entity-anchored approaches

Code is open source: [GitHub link]

7/ If you run a conference or meetup focused on enterprise AI, production ML systems, or operational tools, let's talk.

This solves a real problem every enterprise AI team faces: how to make context reliable and trustworthy.

📧 pete@tamisin.com

---

### 11. CONFERENCE ORGANIZER PITCH (Email Template)

**Subject:** CFP Submission: Deterministic Context-Aware AI for Enterprise Operations

Hi [Organizer Name],

I'm submitting a talk proposal for [Conference Name] on a critical challenge every enterprise AI team faces: how to make context-aware AI reliable and trustworthy for production operations.

**The Problem:**  
Most AI copilots use conversation-based context—they parse chat history to understand references like "this job" or "the expensive one." For enterprise operations, this fragility leads to wrong answers, context drift, and lost trust.

**The Solution:**  
I've architected an alternative approach in production: entity-anchored context, where reports establish available entities, user selections explicitly lock context to specific items (job_123, not "that job"), and all prompts are deterministic functions of that locked, structured context.

**Why This Talk:**
- Addresses real pain: context inference is the #1 trust issue in enterprise AI
- Live demo of production system (not just slides)
- Shows how context flows: reports → selections → graph loading → persistent context → deterministic prompts
- Practical patterns for context architecture that teams can apply immediately
- Challenges the "conversation-first" default with proven alternative

**About Me:**  
I'm Director of Solution Architecture at Capital One Software and former Tech Lead at Databricks, with 20+ years building data and AI platforms for Fortune 500 companies. The system I'll demo is running in production.

**Format Flexibility:**  
Can adapt to 20, 30, or 40-minute slots. Happy to do Q&A, workshop format, or panel discussion on context architecture in AI systems.

Would love to discuss how this fits [Conference Name]'s themes. Full abstract and materials attached.

Best regards,  
Pete Tamisin  
pete@tamisin.com  
[LinkedIn] | [GitHub]

---

### 12. VIDEO RECORDING SCRIPT (3-minute demo for CFP)

**[Title Card: "Deterministic Context-Aware AI: A 3-Minute Demo"]**

**[00:00-00:30] The Problem**

"Hi, I'm Pete Tamisin. Let me show you why conversation-based context fails in enterprise AI—and what to do instead.

Here's a typical context-aware copilot using conversation parsing. I'll look at a cost dashboard, then ask: 'Why is this job so expensive?'

*[Show chat interface giving wrong answer]*

See? It answered about the wrong job. It inferred context from conversation, but got confused. That's the fragility problem."

**[00:30-01:30] The Solution**

"Now watch this. This is deterministic context-aware AI—where context is anchored to entities, not inferred from chat.

*[Pull up Usage Copilot]*

I start with a report showing job costs. When I select a job here...

*[Click job in chart]*

...the system explicitly anchors context to job_123. Behind the scenes, it loads this job's graph context: the job, its cluster, usage patterns, failures.

*[Show context panel]*

This context is now locked. It persists across all my questions about this job. No inference. No ambiguity.

*[Click 'Why is this expensive?']*

When I ask a question, the prompt includes the locked context. Let me show the debug view.

*[Toggle debug panel]*

See? The prompt contains job_123's actual data from the graph. The LLM can't answer about the wrong job."

**[01:30-02:30] Context Persistence**

"Now watch—I'll ask a different question.

*[Click 'How do I optimize this?']*

Same locked context. No re-inference needed. The system knows I'm still talking about job_123 because the context is structurally anchored, not conversationally inferred.

*[Show context inspection]*

I can inspect exactly what context the AI is using. This is what makes it debuggable and trustworthy.

*[Show graph view]*

The context includes the full graph: job relationships, cluster config, usage events. All maintained deterministically."

**[02:30-03:00] The Pitch**

"That's the difference: Entity-anchored context vs conversation-based context.

Reports establish entities. Selections anchor context. Context persists as structured data. Prompts are deterministic functions of that context.

In my full talk, I'll demonstrate this live, share the architecture patterns, and show you how to build context systems that enterprises actually trust.

*[End card: "Pete Tamisin | pete@tamisin.com | GitHub: ChicagoDro"]*

---

### 13. BACKUP MATERIALS CHECKLIST

**Before Conference Day:**

Technical:
- [ ] Usage Copilot running locally (offline mode)
- [ ] Screen recording of full demo (backup if tech fails)
- [ ] PDF export of slides (in case presentation software fails)
- [ ] All architecture diagrams as separate image files
- [ ] GitHub repo link as QR code on closing slide

Content:
- [ ] Printed speaker notes (even if you don't use them)
- [ ] Demo script laminated card (quick reference)
- [ ] Business cards (100+ for networking)
- [ ] One-page handout with key diagrams and GitHub link

Contingency:
- [ ] Video-only version of talk (if live demo fails)
- [ ] Non-demo version of slides (talk about architecture without showing system)
- [ ] Shorter 20-min version (in case of time cuts)
- [ ] Longer 45-min version (if Q&A is included)

---

### 14. POST-TALK AMPLIFICATION PLAN

**Immediately After (Day 1):**
- [ ] Post slides to SlideShare/Speakerdeck
- [ ] Upload video to YouTube (if permitted)
- [ ] LinkedIn post: "Just gave my talk on..."
- [ ] Twitter thread with key points
- [ ] Thank organizers and tag conference

**Week 1:**
- [ ] Write blog post version (expand on key points)
- [ ] Create 3-5 LinkedIn carousels from slide content
- [ ] Engage with attendees who reach out
- [ ] Update GitHub README with "As seen at [Conference]"

**Month 1:**
- [ ] Turn demo into tutorial blog series
- [ ] Submit to more conferences (momentum from first talk)
- [ ] Guest post opportunities on AI/ML blogs
- [ ] Podcast interviews (DM hosts who cover AI)

---

## SUBMISSION CHECKLIST

When submitting to each conference:

- [ ] Short abstract (150 words or less)
- [ ] Long abstract (if required)
- [ ] Speaker bio (match their word count requirements)
- [ ] Headshot (professional, high-res)
- [ ] 3-minute demo video (if allowed)
- [ ] Slide deck outline or sample slides (if requested)
- [ ] Session level (mark as "Intermediate" or "Advanced")
- [ ] Target audience description
- [ ] Session format preference (talk, workshop, panel)
- [ ] Links to GitHub repo and previous talks (if any)
- [ ] A/V requirements (HDMI, audio, internet needs)
- [ ] Travel/accommodation requirements (if applicable)

---

## CONFERENCES TO TARGET (PRIORITIZED)

### Tier 1: Major AI/ML Conferences
1. **AI Engineer Summit** (Oct, SF)
   - CFP: Usually opens April/May
   - Perfect fit: AI engineering focus
   - Audience: 2000+ builders

2. **Applied AI Summit** (Multiple cities)
   - CFP: Rolling
   - Enterprise focus, exact target audience
   - Audience: VPs, Directors, Senior Engineers

3. **MLOps World** (June, Austin)
   - CFP: Usually opens Feb/Mar
   - Production ML angle fits perfectly
   - Audience: 1500+ ML practitioners

### Tier 2: Data/Platform Conferences
4. **Databricks Data + AI Summit** (June, SF)
   - CFP: Opens Jan/Feb
   - You have credibility here as ex-Databricks
   - Audience: 10,000+ data professionals

5. **Data Council** (Multiple cities)
   - CFP: Rolling
   - Strong data engineering audience
   - Audience: 1000+ per event

6. **QCon** (SF, NYC, London)
   - CFP: Opens 6 months prior
   - Software architecture angle
   - Audience: 1500+ senior engineers

### Tier 3: Regional/Accessible Starts
7. **PyData Chicago** (or nearest city)
   - CFP: Usually 3 months before
   - Great for first talk, friendly audience
   - Audience: 200-500

8. **Chicago ML/AI Meetups**
   - Contact organizers directly
   - Practice run before bigger conferences
   - Audience: 50-150

9. **Local Python User Groups**
   - Easy to get speaking slot
   - Build confidence and refine talk
   - Audience: 30-100

---

## FINAL NOTES

**Customization by Conference:**
- For **ML/AI conferences**: Emphasize GraphRAG and RAG architecture
- For **Software conferences**: Emphasize architecture patterns and design principles
- For **Data conferences**: Emphasize operational use cases and data platform angles
- For **Business/Leadership conferences**: Emphasize ROI, trust, and adoption

**Talk Evolution:**
- Start with regional/smaller conferences
- Gather feedback and refine
- Record everything
- Use early talks as credentials for bigger conferences

**Success Metrics:**
- 1-2 conference acceptances in next 3 months
- Video recording of at least one talk
- 500+ views on demo video
- 10+ meaningful connections from talks
- 1-2 companies reaching out about roles

---

## ADDITIONAL CONTEXT: Understanding Entity-Anchored Context Architecture

### How It Works (Technical Detail)

**1. Context Establishment (Report Layer)**
```
Report defines available context entities:
- SQL query defines what's queryable (jobs, clusters, warehouses)
- Each row/bar/point in visualization = potential context anchor
- Report metadata includes entity types and IDs
```

**2. Context Anchoring (Selection Layer)**
```
User clicks entity → System:
1. Captures entity_type (e.g., "job") and entity_id (e.g., "job_123")
2. Loads graph context from knowledge graph:
   - Central node: job_123
   - Connected nodes: cluster, run history, usage events, failures
   - Edges: relationships and metadata
3. Stores in session state as "active context"
```

**3. Context Persistence (Session Layer)**
```
Active context maintained until:
- User explicitly clears it (deselect button)
- User selects different entity (context re-anchors)
- Session ends
```

**4. Prompt Generation (Deterministic Layer)**
```
Every user action generates prompt via:
prompt = template(action_type, active_context)

Example:
- action_type: "diagnose_cost"
- active_context: {
    entity_type: "job",
    entity_id: "job_123",
    graph_data: {job metadata, cluster info, usage patterns}
  }
- Result: Deterministic prompt with all context embedded
```

### Key Architectural Principles

**Principle 1: Context is Explicit**
- No inference from conversation text
- User actions (clicks) are unambiguous signals
- System state is always inspectable

**Principle 2: Context is Structural**
- Graph-based, not text-based
- Relationships are first-class (not just keywords)
- Enables multi-hop reasoning

**Principle 3: Context is Deterministic**
- Same selection = same context = same prompts
- Testable and reproducible
- Version-controllable

**Principle 4: Context is Debuggable**
- Debug panel shows exact context state
- Can inspect graph data being used
- Full audit trail of context changes

### Comparison Table: Context Approaches

| Aspect | Conversation-Based | Entity-Anchored |
|--------|-------------------|-----------------|
| **Context source** | Chat history parsing | Explicit selection + graph |
| **Persistence** | Degrades over time | Stable until cleared |
| **Ambiguity** | "this", "that", "the first one" | entity_id (job_123) |
| **Structure** | Unstructured text | Structured graph data |
| **Determinism** | Same question ≠ same answer | Same selection = same answer |
| **Debuggability** | Hard to inspect | Full inspection tools |
| **Testability** | Flaky (depends on conversation) | Deterministic (repeatable) |
| **Use cases** | Exploration, learning | Operations, analysis, debugging |

### When to Use Each Approach

**Use Entity-Anchored Context When:**
- Working with structured operational data
- Multi-step workflows on specific entities
- High consequence if wrong entity referenced
- Need to maintain context across many questions
- Debugging or troubleshooting scenarios
- Compliance/audit requirements

**Use Conversation-Based Context When:**
- Open-ended exploration
- Learning and education
- Creative brainstorming
- Casual Q&A
- Low stakes interactions
- Natural language is the primary interface

**Hybrid Approach:**
- Start with entity-anchored for precision
- Allow natural language refinements within that context
- Example: Select job_123 → "Tell me more about the cluster it runs on"
  (Context stays anchored to job_123, but allows conversational drill-down)

---

## CRITICAL PATTERN: Grounded Recommendations (Preventing Hallucinations)

### The Hallucination Problem in Recommendations

When users ask "How do I optimize this job?" or "What should I do about these failures?", most AI systems let the LLM generate recommendations from its training data. This leads to:
- **Generic advice** that doesn't apply to your platform
- **Outdated recommendations** based on old best practices
- **Hallucinated procedures** that sound plausible but are wrong
- **Zero accountability** (no way to verify the advice)

### The Solution: Multi-Corpus Grounded Recommendations

Instead of letting the LLM guess, we build a **recommendation corpus** and constrain the LLM to only use it:

**Architecture:**
```
User asks for recommendation
    ↓
System retrieves from 3 corpora:
    1. Telemetry corpus (entity's actual behavior patterns)
    2. Documentation corpus (platform best practices)
    3. Runbook corpus (known procedures for scenarios)
    ↓
Prompt includes:
    - Entity context (job_123 graph data)
    - Retrieved recommendation documents
    - Hard constraint: "Only use provided sources"
    ↓
LLM generates recommendation
    - Every point cites a source document
    - No speculation beyond sources
    - If no runbook exists, system says so
```

### Building the Recommendation Corpora

**1. Platform Documentation Corpus**
```
Sources:
- Official Databricks optimization docs
- Performance tuning guides
- Cost optimization best practices
- Configuration references

Ingestion:
- Scrape/fetch official documentation
- Chunk by section (preserve context)
- Embed with same model as queries
- Index in FAISS/vector store
- Maintain source URLs for citations
```

**2. Runbook Corpus**
```
Sources:
- Markdown files: optimization_runbooks/*.md
- Each runbook covers specific scenario:
  - high_cost_jobs.md
  - spot_eviction_mitigation.md
  - autoscaling_tuning.md
  - shuffle_optimization.md

Format:
---
title: "Optimizing High-Cost Spark Jobs"
scenario: ["high_cost", "performance"]
applies_to: ["job", "cluster"]
---

# Optimization Steps
1. Check partition count (should be 2-3x core count)
2. Review shuffle configuration
3. Consider cluster right-sizing
...

Citations:
- [Databricks Performance Guide](https://docs...)
- [Spark Tuning Documentation](https://spark...)

Ingestion:
- Parse markdown with metadata
- Embed sections separately
- Index with metadata filters (scenario, applies_to)
- Preserve step-by-step structure
```

**3. Telemetry Patterns Corpus** (Future Enhancement)
```
Automatically learned patterns:
- "Jobs with >1000 tasks benefit from larger clusters"
- "This job type sees 30% cost reduction with spot instances"
- "Peak hours: 2pm-4pm EST → autoscaling recommended"

Generated from:
- Historical optimization outcomes
- A/B test results
- Aggregated telemetry analysis
```

### Retrieval Strategy for Recommendations

**Step 1: Hybrid Retrieval**
```python
# Dense vector search for semantic similarity
semantic_docs = vector_search(
    query="optimize high cost job",
    corpus=["docs", "runbooks"],
    top_k=5
)

# Filter by metadata
filtered_docs = filter_by_metadata(
    docs=semantic_docs,
    entity_type="job",
    scenario="high_cost"
)
```

**Step 2: Reranking by Relevance**
```python
# Rerank based on:
# - Entity context match (job characteristics)
# - Recency of runbook
# - Citation quality
# - Past effectiveness (if tracked)

ranked_docs = rerank(filtered_docs, context=job_123_context)
top_recommendations = ranked_docs[:3]
```

**Step 3: Prompt Construction**
```python
prompt = f"""
You are helping optimize {job_123.name}.

ENTITY CONTEXT:
{job_123_graph_data}

AVAILABLE RECOMMENDATIONS (you MUST only use these):
{format_sources(top_recommendations)}

USER QUESTION: How do I optimize this job?

CONSTRAINTS:
- Only recommend actions found in the provided sources
- Cite the source for every recommendation
- If sources don't cover the user's question, say so explicitly
- Do not speculate or add recommendations beyond the sources
- Prioritize runbooks over generic documentation

FORMAT:
1. [Recommendation] - Source: [Document Title, Section]
2. [Recommendation] - Source: [Document Title, Section]
...
"""
```

### Example Output (Grounded Recommendations)

**User:** "How do I optimize this expensive job?"

**System Response:**
```
Based on job_123's usage patterns and our runbooks, here are specific recommendations:

1. **Reduce shuffle operations** - Your job shows 45GB of shuffle data, which is 
   causing spillage. Consider increasing `spark.sql.shuffle.partitions` from 200 
   to 400.
   - Source: [Spark Shuffle Optimization Runbook, Section 2.1]

2. **Enable autoscaling** - This job runs for 2+ hours with varying load. Autoscaling 
   could reduce cost by 20-30% during low-utilization periods.
   - Source: [Databricks Cost Optimization Guide, Autoscaling Best Practices]

3. **Review spot instance ratio** - You're running 100% on-demand. Based on this 
   job's reliability requirements and retry pattern, 70% spot is safe.
   - Source: [High-Cost Job Optimization Runbook, Section 3.4]

Note: I don't have specific runbooks for this job's ML workload characteristics. 
Consider creating a custom runbook based on these optimizations' results.
```

**What makes this trustworthy:**
- Every recommendation cites a source
- No speculation beyond documented procedures
- Acknowledges gaps in knowledge base
- User can verify by reading source documents

### Runbook Versioning & Updates

**Version Control:**
```
runbooks/
  optimization/
    high_cost_jobs_v2.md      (current)
    high_cost_jobs_v1.md      (deprecated)
  reliability/
    spot_mitigation_v3.md     (current)
```

**Update Process:**
1. Engineer writes/updates runbook in markdown
2. Commit to git (version controlled)
3. CI/CD pipeline triggers re-ingestion
4. Embeddings regenerated, index updated
5. Old version marked deprecated but kept for audit

**Benefits:**
- Runbooks evolve with platform changes
- Can A/B test recommendation effectiveness
- Full audit trail of what advice was given when
- Deprecation prevents outdated advice

### Measuring Recommendation Quality

**Metrics to Track:**
```
1. Citation coverage: % of recommendations with citations
2. Runbook usage: Which runbooks are retrieved most
3. User feedback: Thumbs up/down on recommendations
4. Effectiveness: Did recommendation reduce cost/improve reliability?
5. Gap analysis: Questions without matching runbooks
```

**Improvement Loop:**
```
Low citation coverage → LLM is speculating → Add constraint enforcement
High gap analysis → Missing runbooks → Prioritize runbook creation
Low effectiveness → Bad runbooks → Review and update procedures
```

### Production Considerations

**Corpus Freshness:**
- Documentation: Re-scrape weekly (platform docs change)
- Runbooks: Re-index on every git commit
- Telemetry patterns: Regenerate monthly

**Fallback Behavior:**
```python
if no_relevant_runbooks_found:
    response = """
    I don't have specific runbooks for this scenario yet. 
    Based on the platform documentation, here are general best practices:
    [cite docs corpus only]
    
    Recommendation: Create a runbook for this scenario at:
    runbooks/custom/job_{job_type}_optimization.md
    """
```

**Context Window Management:**
- If too many runbooks retrieved, summarize and link
- Prioritize runbooks > docs > telemetry patterns
- Use metadata filters aggressively to reduce retrieval

### Why This Matters for Enterprise Trust

**Traditional LLM recommendations:**
- "Try increasing your cluster size" (vague, generic)
- "Consider using Spark 3.0 features" (might not apply)
- "Optimize your queries" (unhelpful platitude)
- Zero citations, zero accountability

**Grounded recommendations:**
- "Increase spark.sql.shuffle.partitions to 400" (specific, actionable)
- Source: [Runbook: Shuffle Optimization, Section 2.1] (verifiable)
- Acknowledges when no runbook exists (honest about limits)
- Every recommendation traceable to human-written procedure

This is the difference between **AI that sounds smart** and **AI that earns trust**.

---