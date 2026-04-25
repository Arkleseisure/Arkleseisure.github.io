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
          name: "AI-driven pathway",
          description: "The branch where AI raises P(D) and D actually occurs within T. Decomposed below by what kind of AI is responsible.",
          type: "and",
          children: [
            {
              id: "t-d-given-inc",
              name: "D | AI raises P(D)",
              description: "Conditional credence that D occurs within T, given AI raises its probability. Split below by whether the dangerous pathway runs through a single dominant AI or a multipolar one.",
              type: "or",
              children: [
                {
                  id: "t-multi-path",
                  name: "Multipolar AI pathway",
                  description: "The branch where AI raises P(D) via multiple AIs acting at once — coordination failures, race dynamics, or AIs of different types coexisting.",
                  type: "and",
                  children: [
                    {
                      id: "t-multi",
                      name: "Danger comes from multiple AIs",
                      description: "Given AI raises P(D), your credence that the danger comes from multiple AIs rather than a single dominant one. Computed as 1 − P(single dominant AI).",
                      type: "leaf",
                      complement_of: "t-single"
                    },
                    {
                      id: "t-d-multi",
                      name: "D | multipolar AI",
                      description: "Conditional credence that D occurs within T given AI raises P(D) via a multipolar landscape. A multipolar world can mix AIs of different types, so this aggregates across those sub-cases.",
                      type: "leaf"
                    }
                  ]
                },
                {
                  id: "t-single-path",
                  name: "Single dominant AI pathway",
                  description: "The branch where AI raises P(D) via a single dominant AI. Decomposed below by whether the AI has an internal model of D.",
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
                      name: "D | single dominant AI",
                      description: "Conditional credence that D occurs within T, given the danger runs through a single dominant AI. Split below by whether the AI has an internal model of D.",
                      type: "or",
                      children: [
                        {
                          id: "t-rep-path",
                          name: "Internal-model pathway",
                          description: "The branch where the single dominant AI has an internal model of D. Decomposed below by whether the AI expects D.",
                          type: "and",
                          children: [
                            {
                              id: "t-d-given-rep",
                              name: "D | AI has internal model of D",
                              description: "Conditional credence that D occurs within T, given a single dominant AI with an internal model of D. Split below by whether the AI expects D.",
                              type: "or",
                              defaultCollapsed: true,
                              children: [
                                {
                                  id: "t-expects-path",
                                  name: "AI expects D pathway",
                                  description: "The branch where the AI has an internal model of D and expects D to become more likely.",
                                  type: "and",
                                  children: [
                                    {
                                      id: "t-expects",
                                      name: "The AI expects D to become more likely",
                                      description: "Given an AI with an internal model of D, your credence it expects or intends D — the 'deliberate or foreseen' case where the AI's actions are aimed at, or are knowingly consistent with, D.",
                                      type: "leaf"
                                    },
                                    {
                                      id: "t-d-expects",
                                      name: "D | AI expects D",
                                      description: "Conditional credence that D occurs within T given the AI expects D. High values mean the AI's expectation tends to come true; lower values leave room for interventions, containment, or the AI's plans going wrong.",
                                      type: "leaf"
                                    }
                                  ]
                                },
                                {
                                  id: "t-no-expects-path",
                                  name: "AI doesn't expect D pathway",
                                  description: "The branch where the AI has an internal model of D but doesn't expect D to become more likely — D arrives via miscalculation, wrong beliefs, or plans going astray.",
                                  type: "and",
                                  children: [
                                    {
                                      id: "t-no-expects",
                                      name: "The AI doesn't expect D to become more likely",
                                      description: "Given an AI with an internal model of D, your credence it does not expect D — perhaps believing its actions are safe, misjudging consequences, or holding wrong beliefs. Computed as 1 − P(AI expects D).",
                                      type: "leaf",
                                      complement_of: "t-expects"
                                    },
                                    {
                                      id: "t-d-no-expects",
                                      name: "D | AI doesn't expect D",
                                      description: "Conditional credence that D occurs within T, given an AI with an internal model of D that does not expect D — through miscalculation, deception, or plans going wrong.",
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
                          name: "No internal model pathway",
                          description: "The branch where the single dominant AI raises P(D) without representing D as a concept — misaligned optimisation, reward hacking, side-effects, or emergent behaviour.",
                          type: "and",
                          children: [
                            {
                              id: "t-no-rep",
                              name: "The AI has no internal model of D",
                              description: "Given a single dominant AI raising P(D), your credence that it does so without representing D — the 'unaware harm' case. Computed as 1 − P(has internal model of D).",
                              type: "leaf",
                              complement_of: "t-has-rep"
                            },
                            {
                              id: "t-d-no-rep",
                              name: "D | AI has no internal model",
                              description: "Conditional credence that D occurs within T, given a single dominant AI that raises P(D) without an internal model of D.",
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
              description: "Your credence that D is more likely within T in this timeline than in a counterfactual where AI research plateaued before meaningfully affecting D (for example, shortly before 'Attention Is All You Need'). Feel free to substitute your own counterfactual — the question is whether AI, as it actually develops, raises the probability of D.\n\nTechnical note: strictly speaking, 'more likely' is slippery in a Bayesian frame since there's only one real timeline. A cleaner reading: across the distribution of possible worlds similar to ours, is the fraction that undergo D higher among those where AI develops than among those where it doesn't (e.g. worlds where AI research plateaued around or before the first GPT model)?",
              type: "leaf"
            }
          ]
        },
        {
          id: "t-no-inc-path",
          name: "Non-AI pathway",
          description: "The branch where AI doesn't raise P(D) but D still occurs through other causes — the 'base rate' pathway for this worldview.",
          type: "and",
          children: [
            {
              id: "t-no-ai-inc",
              name: "AI doesn't make D more likely",
              description: "AI's development does not raise P(D) relative to the counterfactual. Includes worlds where AI is beneficial or neutral for D, and worlds where AI has little effect either way. Computed as 1 − P(AI makes D more likely).",
              type: "leaf",
              complement_of: "t-ai-inc"
            },
            {
              id: "t-d-no-inc",
              name: "D | AI doesn't raise its probability",
              description: "Conditional credence that D occurs within T, given AI doesn't raise its probability — the base rate of D from non-AI causes (nuclear war, pandemics, natural catastrophes, etc.).",
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
  },

  // =============================================
  // TOY EXAMPLE — minimal illustrative tree for presentations
  // =============================================
  {
    id: "toy-example",
    title: "Toy example",
    substituteNames: false,
    description: "A minimal tree illustrating the core structure: OR root, AND paths, leaf sliders, and a complement pair.",
    tree: {
      id: "toy-root",
      name: "P(E)",
      description: "The probability of event E. Decomposed by conditioning on whether A occurs.",
      type: "or",
      children: [
        {
          id: "toy-a-path",
          name: "E via A",
          description: "P(A) × P(E | A) — the pathway where A occurs and E follows.",
          type: "and",
          children: [
            {
              id: "toy-e-given-a",
              name: "P(E | A)",
              description: "Probability E occurs given A.",
              type: "leaf"
            },
            {
              id: "toy-a",
              name: "P(A)",
              description: "Probability of A.",
              type: "leaf"
            }
          ]
        },
        {
          id: "toy-not-a-path",
          name: "E via ¬A",
          description: "P(¬A) × P(E | ¬A) — the pathway where A doesn't occur but E still does.",
          type: "and",
          children: [
            {
              id: "toy-not-a",
              name: "P(¬A)",
              description: "Complement: 1 − P(A).",
              type: "leaf",
              complement_of: "toy-a"
            },
            {
              id: "toy-e-given-not-a",
              name: "P(E | ¬A)",
              description: "Probability E occurs given A doesn't.",
              type: "leaf"
            }
          ]
        }
      ]
    },
    worldviews: {
      default: {
        name: "Default",
        description: "Neutral 50% priors.",
        probabilities: {
          "toy-a": 0.5,
          "toy-e-given-a": 0.5,
          "toy-e-given-not-a": 0.5
        }
      }
    }
  }
];
