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
  },

  // =============================================
  // AI RISK BY AI TYPE — decomposition by the kind of AI responsible
  // =============================================
  {
    id: "ai-risk-by-type",
    title: "AI risk by AI type",
    description: "Decomposes the probability of D by the kind of AI responsible: whether AI raises P(D) at all, whether the danger comes from a single dominant AI or a multipolar landscape, whether the AI has an internal model of D, and whether it expects D to become more likely.",
    variables: {
      D: { label: "The dangerous event in question", default: "existential catastrophe" },
      T: { label: "Time horizon being considered", default: "30 years" }
    },
    tree: {
      id: "t-root",
      name: "D occurs within T",
      description: "The top-level claim: D occurs within T. Split into two mutually exclusive branches depending on whether AI's development raises the probability of D relative to a counterfactual world without it.",
      type: "or",
      children: [
        {
          id: "t-inc-path",
          name: "AI makes D more likely, and D occurs",
          description: "The branch in which AI's development raises P(D), and D actually occurs within T. Further decomposed by what kind of AI is responsible.",
          type: "and",
          children: [
            {
              id: "t-d-given-inc",
              name: "D occurs, given AI raises its probability",
              description: "Given that AI raises P(D), the conditional probability that D actually occurs within T. Decomposed by whether the danger comes from a single dominant AI or from a multipolar landscape of several AIs acting together.",
              type: "or",
              children: [
                {
                  id: "t-multi-path",
                  name: "Multipolar AI landscape, and D occurs",
                  description: "The sub-branch in which, given AI raises P(D), the dangerous pathway runs through multiple AIs acting simultaneously — coordination failures, race dynamics, AI-on-AI interactions, or several AIs of different types existing at once.",
                  type: "and",
                  children: [
                    {
                      id: "t-d-multi",
                      name: "D | multipolar AI landscape",
                      description: "Given AI raises P(D) via a multipolar landscape, the conditional probability that D occurs within T. Note: a multipolar scenario can involve AIs of several different types simultaneously, so this leaf aggregates across those sub-cases rather than splitting further.",
                      type: "leaf"
                    },
                    {
                      id: "t-multi",
                      name: "Danger comes from multiple AIs",
                      description: "The complement: given AI raises P(D), the dangerous pathway runs through interactions among multiple AIs rather than a single dominant system. Includes AI-on-AI dynamics, race conditions, coordination failures, and worlds with several AIs of different types coexisting. Computed as 1 − P(single dominant AI).",
                      type: "leaf",
                      complement_of: "t-single"
                    }
                  ]
                },
                {
                  id: "t-single-path",
                  name: "Single dominant AI, and D occurs",
                  description: "The sub-branch in which, given AI raises P(D), the dangerous pathway runs through a single dominant AI — and D occurs. Further decomposed by whether that AI has an internal model of D.",
                  type: "and",
                  children: [
                    {
                      id: "t-single",
                      name: "Danger comes from a single dominant AI",
                      description: "Given AI raises P(D), your credence that the dangerous pathway runs through a single dominant AI system — for example, one with a decisive strategic advantage or uniquely capable. The alternative is a multipolar landscape in which several AIs act simultaneously (possibly of different types).",
                      type: "leaf"
                    },
                    {
                      id: "t-d-given-single",
                      name: "D occurs, given a single dominant AI",
                      description: "Given AI raises P(D) via a single dominant AI, the conditional probability that D occurs within T. Decomposed by whether the AI has an internal model of D as a concept.",
                      type: "or",
                      children: [
                        {
                          id: "t-rep-path",
                          name: "AI has internal model of D, and D occurs",
                          description: "The sub-branch in which the single dominant AI has an internal model of D, and D occurs. Further decomposed by whether the AI expects D to become more likely.",
                          type: "and",
                          children: [
                            {
                              id: "t-d-given-rep",
                              name: "D occurs, given AI has an internal model of D",
                              description: "Given a single dominant AI with an internal model of D, the conditional probability that D occurs within T. Decomposed by whether the AI expects or intends D to become more likely.",
                              type: "or",
                              defaultCollapsed: true,
                              children: [
                                {
                                  id: "t-expects-path",
                                  name: "AI expects D, and D occurs",
                                  description: "The sub-branch in which the AI has an internal model of D and expects D to become more likely — and D occurs.",
                                  type: "and",
                                  children: [
                                    {
                                      id: "t-d-expects",
                                      name: "D | AI expects it",
                                      description: "Given an AI that has an internal model of D and expects D to become more likely, the conditional probability that D actually occurs within T. High values correspond to the AI's expectation being reliable; lower values allow for interventions, containment, or the AI's plans going wrong.",
                                      type: "leaf"
                                    },
                                    {
                                      id: "t-expects",
                                      name: "The AI expects D to become more likely",
                                      description: "Given an AI with an internal model of D, your credence that the AI expects or intends D to become more likely. This is the 'deliberate or foreseen' case: the AI is acting in ways it expects will raise P(D), whether because D serves its goals or is a known consequence of its actions.",
                                      type: "leaf"
                                    }
                                  ]
                                },
                                {
                                  id: "t-no-expects-path",
                                  name: "AI doesn't expect D, and D occurs",
                                  description: "The sub-branch in which the AI has an internal model of D but doesn't expect D to become more likely — yet D occurs anyway through miscalculation, wrong beliefs, or plans going astray.",
                                  type: "and",
                                  children: [
                                    {
                                      id: "t-no-expects",
                                      name: "The AI doesn't expect D to become more likely",
                                      description: "The complement: the AI has an internal model of D but does not expect D to become more likely. Covers cases where the AI represents D, believes its actions are safe, misjudges consequences, or holds wrong beliefs about the world. Computed as 1 − P(AI expects D).",
                                      type: "leaf",
                                      complement_of: "t-expects"
                                    },
                                    {
                                      id: "t-d-no-expects",
                                      name: "D | AI doesn't expect it",
                                      description: "Given an AI with an internal model of D that does not expect D to become more likely, the conditional probability that D occurs within T. D materialises despite the AI's beliefs — through miscalculation, deception by others, plans going wrong, or incorrect world-models.",
                                      type: "leaf"
                                    }
                                  ]
                                }
                              ]
                            },
                            {
                              id: "t-has-rep",
                              name: "The AI has an internal model of D",
                              description: "Given a single dominant AI raising P(D), your credence that it has an internal representation of D as a concept — i.e. it 'knows what D is', whether in its world-model, goal specification, or learned features. The alternative is an AI that raises P(D) without representing the danger as such (e.g. via reward hacking, side-effects, or emergent behaviour).",
                              type: "leaf"
                            }
                          ]
                        },
                        {
                          id: "t-no-rep-path",
                          name: "AI has no internal model of D, and D occurs",
                          description: "The sub-branch in which the single dominant AI raises P(D) without having an internal model of D as a concept. Harm from misaligned optimisation, side-effects, reward hacking, or emergent behaviour the AI isn't 'aware' of.",
                          type: "and",
                          children: [
                            {
                              id: "t-no-rep",
                              name: "The AI has no internal model of D",
                              description: "The complement: the single dominant AI raises P(D) without representing D as a concept. This is the 'unaware harm' case — misaligned optimisation, side-effects, reward hacking, mesa-optimisation, or emergent behaviour that produces D without the AI 'knowing' that's what it's doing. Computed as 1 − P(has internal model of D).",
                              type: "leaf",
                              complement_of: "t-has-rep"
                            },
                            {
                              id: "t-d-no-rep",
                              name: "D | AI has no internal model",
                              description: "Given a single dominant AI that raises P(D) without an internal model of D, the conditional probability that D occurs within T.",
                              type: "leaf"
                            }
                          ]
                        }
                      ]
                    }
                  ]
                }
              ]
            },
            {
              id: "t-ai-inc",
              name: "AI makes D more likely",
              description: "Your credence that D is more likely within T in this timeline than in a counterfactual where AI research plateaued before meaningfully affecting D (for example, shortly before 'Attention Is All You Need'). Feel free to substitute your own counterfactual — the question is whether AI, as it actually develops, raises the probability of D.",
              type: "leaf"
            }
          ]
        },
        {
          id: "t-no-inc-path",
          name: "AI doesn't make D more likely, and D occurs",
          description: "The branch in which AI's development doesn't raise P(D) relative to the counterfactual, but D still occurs through other causes — the 'base rate' pathway for this worldview.",
          type: "and",
          children: [
            {
              id: "t-no-ai-inc",
              name: "AI doesn't make D more likely",
              description: "The complement: AI's development does not raise P(D) in this timeline relative to the counterfactual. Includes both worlds where AI is beneficial or neutral for D, and worlds where AI has little effect on D either way. Computed as 1 − P(AI makes D more likely).",
              type: "leaf",
              complement_of: "t-ai-inc"
            },
            {
              id: "t-d-no-inc",
              name: "D happens anyway",
              description: "The conditional probability that D occurs within T, given that AI doesn't raise P(D). This is the base rate of D in the branch where AI isn't contributing — nuclear war, pandemics, natural catastrophes, or other risks unrelated to AI.",
              type: "leaf"
            }
          ]
        }
      ]
    },
    worldviews: {
      pessimistic: {
        name: "High Concern",
        description: "AI very likely raises P(D), and dangerous AIs are expected to be capable and consequential.",
        probabilities: {
          "t-ai-inc": 0.85,
          "t-d-no-inc": 0.1,
          "t-single": 0.6,
          "t-d-multi": 0.4,
          "t-has-rep": 0.7,
          "t-d-no-rep": 0.3,
          "t-expects": 0.3,
          "t-d-expects": 0.8,
          "t-d-no-expects": 0.4
        }
      },
      moderate: {
        name: "Moderate",
        description: "Balanced across all assumptions — default 50% priors.",
        probabilities: {
          "t-ai-inc": 0.5,
          "t-d-no-inc": 0.1,
          "t-single": 0.5,
          "t-d-multi": 0.3,
          "t-has-rep": 0.5,
          "t-d-no-rep": 0.2,
          "t-expects": 0.3,
          "t-d-expects": 0.5,
          "t-d-no-expects": 0.2
        }
      },
      optimistic: {
        name: "Low Concern",
        description: "AI is unlikely to meaningfully raise P(D), and even dangerous AIs rarely cause D to actualise.",
        probabilities: {
          "t-ai-inc": 0.25,
          "t-d-no-inc": 0.03,
          "t-single": 0.4,
          "t-d-multi": 0.1,
          "t-has-rep": 0.7,
          "t-d-no-rep": 0.05,
          "t-expects": 0.05,
          "t-d-expects": 0.6,
          "t-d-no-expects": 0.1
        }
      }
    }
  }
];
