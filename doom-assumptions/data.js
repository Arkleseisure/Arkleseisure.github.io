// Doom Assumptions - Tree Data
// Each tree is a JSON structure with AND/OR/leaf nodes.
// Leaf probabilities are the only free parameters; internal nodes are computed.
//
// Node fields:
//   name        — short display name shown on the card
//   description — longer technical explanation shown in the info panel

const TREES = [
  {
    id: "ai-doom",
    title: "D occurs in the next T years",
    description: "Probability of a dangerous event D (e.g. existential catastrophe) occurring within time horizon T, decomposed by whether a dangerous AI (DAI) is involved.",
    variables: {
      D: { label: "Dangerous event", default: "existential catastrophe" },
      T: { label: "Time horizon", default: "30 years" }
    },
    tree: {
      id: "root",
      name: "D occurs",
      description: "The top-level claim: dangerous event D happens within our time horizon T. Decomposed into AI-caused and non-AI pathways.",
      type: "or",
      children: [
        {
          id: "dai-path",
          name: "AI-caused D",
          description: "D occurs via a dangerous AI system (DAI) being created that leads to the catastrophic outcome. This branch captures the probability that AI is the cause of D.",
          type: "and",
          children: [
            {
              id: "dai-created",
              name: "DAI is created",
              description: "A dangerous AI system — one with the capability and disposition to cause D — is built within the next T years. This decomposes into whether a sufficiently capable AI exists, whether the danger can manifest in time, and whether it actually does.",
              type: "and",
              children: [
                {
                  id: "ai-capable",
                  name: "Capable AI exists",
                  description: "An AI system is built within T years that has sufficient capability to cause dangerous event D, given the right environmental conditions E. This covers both the raw capability question and whether the AI can interact with the environment in the necessary way.",
                  type: "leaf"
                },
                {
                  id: "d-within-time",
                  name: "D possible in time",
                  description: "Given such a capable AI exists, the dangerous event D could unfold within a relevant timeframe t after the AI's creation. This captures whether the causal chain from AI capability to actual harm can play out fast enough.",
                  type: "leaf"
                },
                {
                  id: "d-actualises",
                  name: "D actualises",
                  description: "Given that the conditions for D are met (capable AI exists, timeframe is sufficient), the event actually materialises — i.e. it is not prevented by alignment efforts, safety measures, or other interventions.",
                  type: "leaf"
                }
              ]
            },
            {
              id: "d-given-dai",
              name: "D | DAI created",
              description: "The conditional probability that dangerous event D actually occurs, given that a dangerous AI system has been created. This captures the idea that even if a dangerous AI exists, it may not lead to D (e.g. due to containment, luck, or the AI not pursuing harmful goals).",
              type: "leaf"
            }
          ]
        },
        {
          id: "no-dai-path",
          name: "Non-AI D",
          description: "D occurs through non-AI pathways — for example, bioweapons, nuclear war, natural catastrophe, or other existential risks unrelated to AI. This branch ensures the model accounts for the base rate of D even without AI involvement.",
          type: "and",
          children: [
            {
              id: "no-dai-created",
              name: "No DAI created",
              description: "The complement: no dangerous AI system is built within the time horizon T. Probability is computed as 1 minus P(DAI is created).",
              type: "leaf",
              complement_of: "dai-created"
            },
            {
              id: "d-given-no-dai",
              name: "D | no DAI",
              description: "The conditional probability that dangerous event D occurs through non-AI means, given that no dangerous AI system was created. This is the 'base rate' of existential risk from other sources.",
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
  }
];
