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
    title: "Monopolar catastrophic harm",
    description: "Decomposes the probability of a monopolar catastrophic harm event, splitting by whether a dangerous AI system is involved.",
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
  }
];
