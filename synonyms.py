# synonyms.py

import unicodedata, re

def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s
    
# Groups of equivalent terms (bidirectional synonyms)
SYNONYM_GROUPS = [
    #   Languages  
    ["english c1", "fluent english", "advanced english", "english professional", "Speaks english professional", "proficient english"],
    ["native english", "mother tongue english", "english native", "inglés nativo"],

    #   Tools & Skills  
    ["power bi", "microsoft power bi", "powerbi"],
    ["sql", "structured query language"],
    ["excel advanced", "advanced excel", "excel experto", "microsoft excel avanzado"],

    #   Renewable Energy Technologies  
    ["wind turbine", "wind energy", "onshore wind", "offshore wind", "energía eólica", "aerogenerador"],
    ["solar pv", "photovoltaic", "solar panels", "solar energy", "energía solar", "solar fotovoltaica", "fotovoltaico"],
    ["hydropower", "hydroelectric", "hydro energy", "energía hidroeléctrica", "hidroeléctrico"],
    ["biomass", "bioenergy", "energía de biomasa", "bioenergía"],
    ["green hydrogen", "renewable hydrogen", "h2 green hydrogen", "hidrógeno verde", "hidrógeno renovable"],

    #   Contracts & Markets  
    ["ppa", "power purchase agreement", "ppas", "contrato ppa", "acuerdo de compra de energía"],
    ["lcoe", "levelized cost of energy", "levelised cost of electricity", "coste nivelado de energía"],

    #   Job Titles / Roles  
    ["head of controlling", "controlling manager", "controlling lead", "finance controlling head", "jefe de controlling", "gerente de controlling"],
    ["project manager", "program manager", "project lead", "jefe de proyectos", "gerente de proyectos"],
    ["business development manager", "bd manager", "commercial manager", "desarrollo de negocio", "gerente comercial"],

    #   Finance & Investment  
    ["financial modelling", "financial modeling", "modelo financiero", "modelización financiera"],
    ["capex", "capital expenditure", "gasto de capital", "inversión en activos"],
    ["opex", "operational expenditure", "operating costs", "gasto operativo", "costes operativos"],

    #   Regulations & Strategy  
    ["esg", "environmental social governance", "sustainability reporting", "criterios esg", "gobernanza ambiental social"],
    ["carbon credits", "co2 certificates", "emission allowances", "créditos de carbono", "certificados de co2", "derechos de emisión"],
    ["fit", "feed-in tariff", "tarifa regulada", "tarifa feed-in"],
    ["renewable energy directive", "eu red ii", "red ii", "directiva de energías renovables"],

       #   Finance & Controlling  
    ["fp&a", "financial planning & analysis", "fpna"],
    ["financial controller", "finance controller", "head of controlling"],
    ["internal control", "internal controls", "control culture"],
    ["management control", "controlling", "management controlling"],
    ["budgeting", "budget preparation", "budget planning"],
    ["forecasting", "financial forecasting", "financial projections"],
    ["financial reporting", "reporting", "finance reporting"],
    ["audit", "auditing", "auditor", "external audit", "internal audit"],
    ["compliance", "regulatory compliance", "governance compliance"],
    ["risk management", "enterprise risk management", "risk analysis"],
    ["cost optimization", "cost efficiency", "cost reduction"],
    ["decision making", "business decision making"],

    #   Job Titles / Roles  
    ["finance manager", "financial manager"],
    ["deputy manager", "deputy director", "assistant manager"],
    ["administration and finance manager", "finance and administration manager"],
    ["senior consultant", "consulting senior", "consultant"],
    ["audit manager", "manager audit", "audit lead"],
    ["internal audit supervisor", "audit supervisor"],
    ["business intelligence", "bi", "business insights"],

    #   Skills / Soft Skills  
    ["leadership", "team leadership", "situational leadership"],
    ["team management", "people management", "managing teams"],
    ["negotiation", "stakeholder negotiation"],
    ["communication skills", "effective communication", "strong communication"],
    ["strategic mindset", "strategic thinking"],
    ["results oriented", "results-driven", "results focused"],

    #   Reporting & International Standards  
    ["international reporting", "global reporting"],
    ["us gaap", "gaap"],
    ["ifrs", "international financial reporting standards"],

    #   Consulting / Advisory  
    ["consulting", "advisory"],
    ["assurance", "risk assurance"],
    ["governance risk compliance", "grc", "risk & compliance"],

    #   Education & Qualifications  
    ["commercial engineer", "business engineer"],
    ["management control diploma", "diploma in management control"],

    #   Tech Tools (finance & BI)  
    ["erp", "enterprise resource planning"],
    ["sap", "sap hana", "sap erp"],
    ["tableau", "tableau software"],
    ["power bi", "microsoft power bi", "powerbi"],
    ["sql", "structured query language"],
    ["excel advanced", "advanced excel", "excel expert", "microsoft excel advanced"],

    #   General Business Terms  
    ["cash flow", "working capital"],
    ["closing process", "month-end closing", "financial closing"],
    ["tax compliance", "tax reporting", "tax management"],
    ["inventory control", "stock control"],
    ["salary advances", "pay advances"],

    #   Renewable Energy & Sustainability  
    ["wind turbine", "wind energy", "onshore wind", "offshore wind"],
    ["solar pv", "photovoltaic", "solar panels", "solar energy"],
    ["hydropower", "hydroelectric", "hydro energy"],
    ["biomass", "bioenergy"],
    ["green hydrogen", "renewable hydrogen", "h2 hydrogen"],
    ["ppa", "power purchase agreement", "ppas"],
    ["lcoe", "levelized cost of energy", "levelized cost of electricity"],
    ["esg", "environmental social governance", "sustainability reporting"],
    ["carbon credits", "carbon trading", "emission allowances"],
    ["feed-in tariff", "fit"],
    ["renewable energy directive", "red ii", "eu renewable directive"],

    #   Reporting & Analysis  
    ["reporting & analysis", "financial analysis", "business analysis", "management reports"],
    ["revenue forecasting", "sales forecasting", "demand forecasting"],
    ["profitability reports", "margin analysis", "profitability analysis"],
    ["p&l", "profit and loss", "income statement"],

    #   Forecasting & Budgeting  
    ["forecast", "financial forecast", "revenue forecast"],
    ["budget", "budgeting", "annual budget", "budget planning"],

    #   Pricing & Revenue Management  
    ["pricing", "pricing strategy", "price management"],
    ["revenue management", "margin management", "profit optimization"],
    ["downstream pricing", "retail fuel pricing", "fuel pricing strategy"],
    ["pricing analyst", "pricing specialist", "pricing manager"],

    #   Consolidation & International  
    ["consolidation", "financial consolidation", "regional consolidation"],
    ["headquarters reporting", "corporate reporting"],
    ["south america consolidation", "regional consolidation"],
    ["international reporting", "global reporting"],

    #   Capex & Opex  
    ["capex", "capital expenditure", "investment projects"],
    ["opex", "operational expenditure", "operating costs"],
    ["investment project evaluation", "capex request analysis"],

    #   Operations & Control  
    ["internal control", "operational controls", "preventive controls"],
    ["inventory obsolescence", "slow moving inventory", "obsolete stock"],
    ["inventory control", "stock management", "inventory management"],

    #   Leadership & People Management  
    ["team leadership", "people management", "staff management"],
    ["led teams", "team development", "managing employees"],
    ["customer service", "client relations"],

    #   Energy & Fuel Sector  
    ["downstream", "retail fuel", "fuel stations", "service stations"],
    ["convenience store sales", "c-store sales"],
    ["petrobras", "oil & gas downstream", "fuel retail"],
    ["air products", "industrial gases", "specialty gases"],

    #   Tools & Systems  
    ["flexline", "inventory system", "billing system"],
    ["dashboard", "business dashboard", "performance dashboard"],

    #   General Finance & Accounting  
    ["cash flow", "liquidity management", "working capital"],
    ["accounting", "general accounting", "financial accounting"],
    ["payroll", "compensation & benefits"],
    ["hr administration", "hr support", "administrative hr"],

    #   Education / Certifications  
    ["master in financial management", "financial management master's"],
    ["master in pricing & revenue management", "pricing & revenue management master's"],
    ["certified public accountant", "cpa"],
    ["business administration", "bachelor in business administration"],
]


# Build dict for expansion
SYNONYMS = {}
for group in SYNONYM_GROUPS:
    for term in group:
        key = _normalize(term)
        SYNONYMS[key] = [t for t in group if _normalize(t) != key]

def expand_with_synonyms(text: str) -> str:
    """
    STRICT (exact phrase) expansion:
    - Only expand when the WHOLE input phrase exactly matches a key (after normalization).
    - No substring/contains logic. No partial matches.
    """
    base = (text or "").strip()
    key = _normalize(base)
    syns = SYNONYMS.get(key, [])
    if not syns:
        return base  # no expansion if not an exact match
    # de-dup while preserving order, and avoid re-adding the original phrase
    seen = {key}
    add = []
    for s in syns:
        n = _normalize(s)
        if n not in seen:
            seen.add(n)
            add.append(s)
    return base + (" " + " ".join(add) if add else "")
