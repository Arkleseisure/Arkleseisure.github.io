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
    description: "Formal decomposition using the DAI(D, E, T) framework. A DAI is defined as: an AI system created within the next T years such that D can co-occur in environment E within T years. This avoids causation questions — it is about conditional co-occurrence, not whether the AI caused D.",
    variables: {
      D: { label: "Danger event", default: "existential catastrophe" },
      T: { label: "Time horizon from now", default: "30 years" },
      E: { label: "Environment / conditions", default: "current geopolitical conditions" }
    },
    tree: {
      id: "f-root",
      name: "P(D in T)",
      description: "The probability that danger D occurs within timeframe T from now. Decomposed via the law of total probability by conditioning on whether a DAI(D, E, T) is created.\n\nP(D in T) = P(D in T | DAI created) \u00d7 P(DAI created) + P(D in T | \u00acDAI) \u00d7 P(\u00acDAI)",
      type: "or",
      children: [
        {
          id: "f-dai-path",
          name: "D via DAI pathway",
          description: "The joint probability that danger D occurs within T AND a DAI(D, E, T) is created. Equals P(D in T | DAI created) \u00d7 P(DAI created).\n\nThis branch captures the AI-mediated pathway to D. A DAI(D, E, T) is an AI created within the next T years such that D can co-occur in environment E within T years. The definition is about co-occurrence, not causation.",
          type: "and",
          children: [
            {
              id: "f-d-given-dai",
              name: "P(D | DAI exists)",
              description: "The conditional probability that D actually occurs within T, given that a DAI(D, E, T) has been created.\n\nA DAI is defined as an AI where D CAN co-occur in E within T — but that doesn't guarantee D DOES occur. This conditional captures the gap between possibility and actuality:\n\u2022 The actual environment may differ from the assumed E\n\u2022 Interventions after creation (shutdown, containment) could prevent D\n\u2022 D co-occurring with the AI in E is possible but not certain\n\u2022 Multiple things may need to go wrong simultaneously",
              type: "leaf"
            },
            {
              id: "f-dai-created",
              name: "P(DAI(D,E,T) created)",
              description: "The probability that a DAI(D, E, T) is created — an AI system built within the next T years such that D can co-occur in environment E within T years.\n\nDecomposed into two mutually exclusive pathways based on whether the AI is causally responsible for D:\n1. Non-causal DAI: D would happen anyway, and an AI co-occurs with it\n2. Causal DAI: D would not happen without AI, but an AI is created that makes it happen",
              type: "or",
              children: [
                {
                  id: "f-noncausal-dai",
                  name: "Non-causal DAI",
                  description: "A DAI exists, but the AI is not causally responsible for D. D would have occurred in E within T regardless of the AI's existence, and an AI system gets created which does not change this.\n\nThis pathway captures scenarios where:\n\u2022 D is already likely (e.g. nuclear war, pandemic) independent of AI\n\u2022 An AI system happens to exist during the same period\n\u2022 The AI's presence does not materially affect whether D occurs\n\nThe DAI definition counts these because it is based on co-occurrence, not causation.",
                  type: "and",
                  children: [
                    {
                      id: "f-d-without-ai",
                      name: "P(D in E, T | no AI)",
                      description: "The probability that D can occur in environment E within T years even without any AI system being involved.\n\nThis captures the base rate of D from non-AI causes in environment E. If D is an existential catastrophe, this includes risks from:\n\u2022 Nuclear war\n\u2022 Engineered pandemics\n\u2022 Natural catastrophes\n\u2022 Other non-AI existential risks\n\nA high value means D is likely regardless of AI; a low value means D is primarily an AI-related risk.",
                      type: "leaf"
                    },
                    {
                      id: "f-ai-no-change",
                      name: "P(AI created, D unchanged)",
                      description: "Given that D can already occur in E within T without AI: the probability that an AI system gets created which does not change this fact.\n\nThis asks: does an AI get built that doesn't prevent or accelerate D? This could be high if:\n\u2022 AI systems are focused on other domains and don't affect D\n\u2022 AI safety measures don't specifically address D\n\u2022 The AI is not powerful enough to prevent or cause D\n\nNote: this is specifically about AI systems that don't change D's probability, not about whether any AI exists.",
                      type: "leaf"
                    }
                  ]
                },
                {
                  id: "f-causal-dai",
                  name: "Causal DAI",
                  description: "A DAI exists because the AI is causally responsible for D. D would NOT have occurred in E within T without AI, but an AI system is created that changes this — making D possible.\n\nThis pathway captures the core AI risk scenario:\n\u2022 D is not likely on its own within the timeframe\n\u2022 An AI system is built that creates new pathways to D\n\u2022 The AI's capabilities, goals, or effects enable D to occur\n\nThis is the pathway most AI safety work focuses on preventing.",
                  type: "and",
                  children: [
                    {
                      id: "f-d-not-without-ai",
                      name: "P(\u00acD in E, T | no AI)",
                      description: "The probability that D would NOT occur in E within T without AI. This is the complement of P(D in E, T | no AI).\n\nComputed as: 1 \u2212 P(D in E, T | no AI)\n\nA high value means D is primarily an AI-related risk — it wouldn't happen on its own. A low value means D is likely regardless, so the causal DAI pathway contributes less.",
                      type: "leaf",
                      complement_of: "f-d-without-ai"
                    },
                    {
                      id: "f-ai-causes-d",
                      name: "P(AI created, enables D)",
                      description: "Given that D would not occur in E within T without AI: the probability that an AI system is created which changes this — making D possible.\n\nThis is the central question of AI risk: does an AI get built that enables a danger which wouldn't otherwise exist? This depends on:\n\u2022 Whether sufficiently capable AI is built within T\n\u2022 Whether that AI has properties (goals, capabilities, deployment context) that create pathways to D\n\u2022 Whether safety measures, alignment, or regulation fail to prevent this\n\nThis leaf is where most disagreement between AI risk worldviews concentrates.",
                      type: "leaf"
                    }
                  ]
                }
              ]
            }
          ]
        },
        {
          id: "f-no-dai-path",
          name: "D without DAI pathway",
          description: "The joint probability that D occurs within T AND no DAI(D, E, T) is created. This is the non-AI pathway.\n\nEquals P(D in T | \u00acDAI created) \u00d7 P(\u00acDAI created).\n\nThis branch accounts for the base rate of D from non-AI sources: bioweapons, nuclear war, pandemics, natural catastrophes, or other risks.",
          type: "and",
          children: [
            {
              id: "f-no-dai",
              name: "P(\u00acDAI in T)",
              description: "The probability that no DAI(D, E, T) is created within T. This is the complement of P(DAI created).\n\nComputed as: 1 \u2212 P(DAI(D,E,T) created)",
              type: "leaf",
              complement_of: "f-dai-created"
            },
            {
              id: "f-d-given-no-dai",
              name: "P(D | \u00acDAI)",
              description: "The conditional probability that D occurs within T through non-AI pathways, given that no DAI(D, E, T) is created.\n\nThis is the base rate of danger D from other sources. In worlds where no AI system meeting the DAI definition is built, what is the probability of D from other causes?\n\nFor D = existential catastrophe, this covers:\n\u2022 Engineered pandemics / bioweapons\n\u2022 Nuclear war\n\u2022 Supervolcanic eruption or asteroid impact\n\u2022 Other unforeseen existential risks",
              type: "leaf"
            }
          ]
        }
      ]
    },
    worldviews: {
      pessimistic: {
        name: "High Concern",
        description: "Assigns high probability to AI-driven existential risk. Believes D is somewhat likely even without AI, and AI significantly increases the risk.",
        probabilities: {
          "f-d-without-ai": 0.1,
          "f-ai-no-change": 0.8,
          "f-ai-causes-d": 0.5,
          "f-d-given-dai": 0.75,
          "f-d-given-no-dai": 0.05
        }
      },
      moderate: {
        name: "Moderate",
        description: "Balanced assessment. Non-AI risk of D is low, moderate chance that AI creates new pathways to D.",
        probabilities: {
          "f-d-without-ai": 0.05,
          "f-ai-no-change": 0.7,
          "f-ai-causes-d": 0.25,
          "f-d-given-dai": 0.5,
          "f-d-given-no-dai": 0.03
        }
      },
      optimistic: {
        name: "Low Concern",
        description: "Believes D is very unlikely without AI, and AI is unlikely to create new pathways to D. Even if a DAI exists, D probably won't actually occur.",
        probabilities: {
          "f-d-without-ai": 0.02,
          "f-ai-no-change": 0.6,
          "f-ai-causes-d": 0.1,
          "f-d-given-dai": 0.15,
          "f-d-given-no-dai": 0.02
        }
      }
    }
  }
];
