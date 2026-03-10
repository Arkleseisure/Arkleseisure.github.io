// Doom Assumptions - Tree Data
// Each tree is a JSON structure with AND/OR/leaf nodes.
// Leaf probabilities are the only free parameters; internal nodes are computed.
//
// Node fields:
//   name        — short display name shown on the card
//   description — longer technical explanation shown in the info panel

const TREES = [
  // =============================================
  // CASUAL VERSION — accessible language
  // =============================================
  {
    id: "ai-doom-casual",
    title: "Monopolar catastrophic harm (casual)",
    description: "Decomposes the probability of a monopolar catastrophic harm event, splitting by whether a dangerous AI system is involved. Uses accessible language.",
    variables: {
      Danger: { label: "The dangerous event in question", default: "existential catastrophe" },
      Timeframe: { label: "Time horizon being considered", default: "30 years" }
    },
    tree: {
      id: "root",
      name: "Danger occurs within Timeframe",
      description: "The top-level claim: the specified danger occurs within the given timeframe. Decomposed into AI-caused and non-AI pathways.",
      type: "or",
      children: [
        {
          id: "dai-path",
          name: "AI-caused Danger",
          description: "Danger occurs via a dangerous AI system (DAI) being created that leads to the catastrophic outcome. This branch captures the probability that AI is the cause.",
          type: "and",
          children: [
            {
              id: "d-given-dai",
              name: "Danger | DAI created",
              description: "The conditional probability that the Danger actually occurs, given that a dangerous AI system has been created. This captures the idea that even if a dangerous AI exists, it may not lead to the Danger (e.g. due to containment, luck, or the AI not pursuing harmful goals).",
              type: "leaf"
            },
            {
              id: "dai-created",
              name: "DAI is created",
              description: "A dangerous AI system — one with the capability and disposition to cause the specified Danger — is built within the Timeframe. This decomposes into whether a sufficiently capable AI exists, whether the Danger can manifest in time, and whether it actually does.",
              type: "and",
              children: [
                {
                  id: "ai-capable",
                  name: "Capable AI exists",
                  description: "An AI system is built within the Timeframe that has sufficient capability to cause the specified Danger, given the right environmental conditions. This covers both the raw capability question and whether the AI can interact with the environment in the necessary way.",
                  type: "leaf"
                },
                {
                  id: "d-within-time",
                  name: "Danger possible before Timeframe ends",
                  description: "Given such a capable AI exists, the Danger could unfold in the remaining time between the AI's creation and the end of the Timeframe. This captures whether the causal chain from AI capability to actual harm can play out before the time horizon is reached.",
                  type: "leaf"
                },
                {
                  id: "d-actualises",
                  name: "Danger actualises",
                  description: "Given that the conditions for Danger are met (capable AI exists, timeframe is sufficient), the event actually materialises — i.e. it is not prevented by alignment efforts, safety measures, or other interventions.",
                  type: "leaf"
                }
              ]
            }
          ]
        },
        {
          id: "no-dai-path",
          name: "Non-AI Danger",
          description: "Danger occurs through non-AI pathways — for example, bioweapons, nuclear war, natural catastrophe, or other risks unrelated to AI. This branch ensures the model accounts for the base rate of Danger even without AI involvement.",
          type: "and",
          children: [
            {
              id: "no-dai-created",
              name: "No DAI created",
              description: "The complement: no dangerous AI system is built within the Timeframe. Probability is computed as 1 minus P(DAI is created).",
              type: "leaf",
              complement_of: "dai-created"
            },
            {
              id: "d-given-no-dai",
              name: "Danger | no DAI",
              description: "The conditional probability that the Danger occurs through non-AI means, given that no dangerous AI system was created. This is the 'base rate' of the Danger from other sources.",
              type: "leaf"
            }
          ]
        }
      ]
    },
    worldviews: {
      pessimistic: {
        name: "High Concern",
        description: "Assigns high probability to AI-driven existential risk.",
        probabilities: {
          "ai-capable": 0.8,
          "d-within-time": 0.7,
          "d-actualises": 0.6,
          "d-given-dai": 0.7,
          "d-given-no-dai": 0.05
        }
      },
      moderate: {
        name: "Moderate",
        description: "Balanced view acknowledging both risks and the difficulty of causing extinction.",
        probabilities: {
          "ai-capable": 0.5,
          "d-within-time": 0.5,
          "d-actualises": 0.5,
          "d-given-dai": 0.5,
          "d-given-no-dai": 0.05
        }
      },
      optimistic: {
        name: "Low Concern",
        description: "Assigns low probability — alignment is tractable and AI won't cause catastrophe.",
        probabilities: {
          "ai-capable": 0.3,
          "d-within-time": 0.2,
          "d-actualises": 0.15,
          "d-given-dai": 0.2,
          "d-given-no-dai": 0.02
        }
      }
    }
  },

  // =============================================
  // FORMAL VERSION — precise DAI(p, t, D, E) notation
  // =============================================
  {
    id: "ai-doom-formal",
    title: "Monopolar catastrophic harm (formal)",
    substituteNames: false,
    cssClass: "formal",
    description: "Formal decomposition using the DAI(p, t, D, E) framework. A DAI is defined as: an AI system such that, when created in environment E, danger D occurs within time t with probability at least p. This avoids causation questions by framing everything as conditional probability.",
    variables: {
      D: { label: "Danger event", default: "existential catastrophe" },
      T: { label: "Time horizon from now", default: "30 years" },
      t: { label: "Time window after AI creation", default: "10 years" },
      E: { label: "Environment / conditions", default: "current geopolitical conditions" },
      p: { label: "Probability threshold for DAI classification", default: "0.5" }
    },
    tree: {
      id: "f-root",
      name: "P(D in T)",
      description: "The probability that danger D occurs within timeframe T from now. Decomposed via the law of total probability by conditioning on whether a DAI(p, t, D, E) is created within T.\n\nP(D in T) = P(D in T | DAI created) \u00d7 P(DAI created) + P(D in T | \u00acDAI) \u00d7 P(\u00acDAI)",
      type: "or",
      children: [
        {
          id: "f-dai-path",
          name: "D via DAI pathway",
          description: "The joint probability that danger D occurs within T AND a DAI(p, t, D, E) is created within T. Equals P(D in T | DAI created) \u00d7 P(DAI created in T).\n\nThis branch captures the AI-mediated pathway to D. Note: we use conditional co-occurrence, not causation — the DAI definition only requires that D occurs given the AI's presence, not that the AI caused D.",
          type: "and",
          children: [
            {
              id: "f-d-given-dai",
              name: "P(D | DAI exists)",
              description: "The conditional probability that D occurs within T, given that a DAI(p, t, D, E) has been created.\n\nBy the definition of DAI, we know that in environment E, D occurs within time t with probability \u2265 p. However, this conditional probability may differ from p because:\n\u2022 The actual environment at deployment may differ from E\n\u2022 T is measured from now, not from AI creation — the AI may be created late in the timeframe\n\u2022 Interventions after creation (shutdown, containment) could prevent D\n\u2022 The probability p is a lower bound, not an exact value\n\nThis leaf captures these residual uncertainties.",
              type: "leaf"
            },
            {
              id: "f-dai-created",
              name: "P(DAI(p,t,D,E) in T)",
              description: "The probability that a DAI(p, t, D, E) — an AI such that D occurs within t with probability \u2265 p when created in environment E — is created within timeframe T.\n\nDecomposed into three factors:\n1. Does an AI get created that co-occurs with D in E?\n2. Can D happen within time t of that AI's creation?\n3. Is the conditional probability \u2265 p?",
              type: "and",
              children: [
                {
                  id: "f-ai-cooccur",
                  name: "P(AI cooccurs with D in E)",
                  description: "The probability that an AI system is created within T such that it co-occurs with danger D in environment E.\n\nThis asks: is there an AI system AND an environment E in which D can happen while this AI exists? This is deliberately NOT asking whether the AI causes D — only whether D and the AI can coexist in environment E.\n\nThis captures:\n\u2022 Whether sufficiently advanced AI is built within T\n\u2022 Whether the deployment environment matches conditions E\n\u2022 Whether D is physically possible in the presence of this AI",
                  type: "leaf"
                },
                {
                  id: "f-d-in-t",
                  name: "P(D within t | AI, E)",
                  description: "Given that an AI system exists that co-occurs with D in environment E: the probability that D can occur within time t after the AI's creation.\n\nThis captures the temporal constraint. Even if an AI and D can coexist in E, the danger might take longer than t to materialise. For example:\n\u2022 If t = 10 years and the AI needs decades to accumulate enough influence for D to occur, this probability is low\n\u2022 If D could happen rapidly once the AI exists (e.g. a fast takeoff scenario), this probability is high\n\nNote: t is measured from AI creation, distinct from T which is measured from now.",
                  type: "leaf"
                },
                {
                  id: "f-prob-threshold",
                  name: "P(prob \u2265 p)",
                  description: "Given that an AI co-occurs with D in E and D can occur within time t: the probability that this actually constitutes a DAI — i.e., that the conditional probability of D occurring meets or exceeds the threshold p.\n\nThis is the definitional threshold. A low p (e.g. 0.1) means we count AI systems that only slightly elevate the risk of D. A high p (e.g. 0.9) means we only count AI systems that make D near-certain.\n\nThe choice of p reflects how strictly we define 'dangerous'. With p = 0.5, we count an AI as a DAI if D is more likely than not to occur within t when the AI is created in E.",
                  type: "leaf"
                }
              ]
            }
          ]
        },
        {
          id: "f-no-dai-path",
          name: "D without DAI pathway",
          description: "The joint probability that D occurs within T AND no DAI(p, t, D, E) is created within T. This is the non-AI pathway.\n\nEquals P(D in T | \u00acDAI created) \u00d7 P(\u00acDAI created in T).\n\nThis branch accounts for the base rate of D from non-AI sources: bioweapons, nuclear war, pandemics, natural catastrophes, or other existential risks. It ensures the model doesn't attribute all risk to AI.",
          type: "and",
          children: [
            {
              id: "f-no-dai",
              name: "P(\u00acDAI in T)",
              description: "The probability that no DAI(p, t, D, E) is created within T. This is the complement of P(DAI created in T).\n\nComputed as: 1 \u2212 P(DAI(p,t,D,E) created in T)",
              type: "leaf",
              complement_of: "f-dai-created"
            },
            {
              id: "f-d-given-no-dai",
              name: "P(D | \u00acDAI)",
              description: "The conditional probability that D occurs within T through non-AI pathways, given that no DAI(p, t, D, E) is created.\n\nThis is the 'base rate' of danger D from other sources. Note that this is conditional on no DAI existing — in worlds where advanced AI is not built (or is built safely), what is the probability of D from other causes?\n\nFor D = existential catastrophe, this covers risks like:\n\u2022 Engineered pandemics / bioweapons\n\u2022 Nuclear war\n\u2022 Supervolcanic eruption or asteroid impact\n\u2022 Other unforeseen existential risks",
              type: "leaf"
            }
          ]
        }
      ]
    },
    worldviews: {
      pessimistic: {
        name: "High Concern",
        description: "Assigns high probability to AI-driven existential risk. Believes advanced AI is likely, DAI threshold is easily met, and non-AI risks are non-trivial.",
        probabilities: {
          "f-ai-cooccur": 0.85,
          "f-d-in-t": 0.7,
          "f-prob-threshold": 0.6,
          "f-d-given-dai": 0.75,
          "f-d-given-no-dai": 0.05
        }
      },
      moderate: {
        name: "Moderate",
        description: "Balanced assessment. Advanced AI is plausible within the timeframe, but significant uncertainty about whether it constitutes a DAI and whether D follows.",
        probabilities: {
          "f-ai-cooccur": 0.55,
          "f-d-in-t": 0.5,
          "f-prob-threshold": 0.45,
          "f-d-given-dai": 0.5,
          "f-d-given-no-dai": 0.03
        }
      },
      optimistic: {
        name: "Low Concern",
        description: "Believes alignment is tractable, the DAI threshold is hard to meet, and even if a DAI exists the probability of D is low due to safeguards.",
        probabilities: {
          "f-ai-cooccur": 0.35,
          "f-d-in-t": 0.25,
          "f-prob-threshold": 0.2,
          "f-d-given-dai": 0.15,
          "f-d-given-no-dai": 0.02
        }
      }
    }
  }
];
