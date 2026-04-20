// Doom Assumptions - Tree Data
// Each tree is a JSON structure with AND/OR/leaf nodes.
// Leaf probabilities are the only free parameters; internal nodes are computed.
//
// Node fields:
//   name        — short display name shown on the card
//   description — longer technical explanation shown in the info panel

const TREES = [
  // =============================================
  // AI RISK BY AI TYPE — decomposition by the kind of AI responsible
  // =============================================
  {
    id: "ai-risk-by-type",
    title: "AI risk by AI type",
    description: "Decomposes the probability of D by the kind of AI responsible: whether AI raises P(D) at all, whether the danger comes from a single dominant AI or a multipolar landscape, whether the AI has an internal model of D, and whether it expects D to become more likely.",
    variables: {
      D: { displayName: "Danger", label: "The dangerous event in question", default: "existential catastrophe" },
      T: { displayName: "Timeframe", label: "Time horizon being considered", default: "30 years" }
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
