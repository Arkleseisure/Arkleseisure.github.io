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
      description: "The top-level claim: D occurs within T. Split into two mutually exclusive sets of worlds depending on whether AI's development raises the probability of D relative to a counterfactual without it.",
      type: "or",
      children: [
        {
          id: "t-inc-path",
          name: "AI-driven worlds",
          description: "Worlds where AI raises P(D) and D actually occurs within T. Decomposed below by what kind of AI is responsible.",
          type: "and",
          children: [
            {
              id: "t-d-given-inc",
              name: "D | AI raises P(D)",
              description: "Among worlds where AI raises P(D), your credence that D occurs within T. Split below by whether the danger runs through a single dominant AI or a multipolar one.",
              type: "or",
              children: [
                {
                  id: "t-multi-path",
                  name: "Multipolar AI worlds",
                  description: "Worlds where AI raises P(D) via multiple AIs acting at once — coordination failures, race dynamics, or AIs of different types coexisting.",
                  type: "and",
                  children: [
                    {
                      id: "t-multi",
                      name: "Danger comes from multiple AIs",
                      description: "Among worlds where AI raises P(D), your credence that the danger comes from multiple AIs rather than a single dominant one. Computed as 1 − P(single dominant AI).",
                      type: "leaf",
                      complement_of: "t-single"
                    },
                    {
                      id: "t-d-multi",
                      name: "D | multipolar AI",
                      description: "Among worlds where AI raises P(D) via a multipolar landscape, your credence that D occurs within T. A multipolar world can mix AIs of different types, so this aggregates across those sub-cases.",
                      type: "leaf"
                    }
                  ]
                },
                {
                  id: "t-single-path",
                  name: "Single dominant AI worlds",
                  description: "Worlds where AI raises P(D) via a single dominant AI. Decomposed below by whether the AI has an internal model of D.",
                  type: "and",
                  children: [
                    {
                      id: "t-single",
                      name: "Danger comes from a single dominant AI",
                      description: "Among worlds where AI raises P(D), your credence that the danger runs through a single dominant AI — for example, one with a decisive strategic advantage or uniquely capable. The alternative is a multipolar landscape in which several AIs act simultaneously (possibly of different types).",
                      type: "leaf"
                    },
                    {
                      id: "t-d-given-single",
                      name: "D | single dominant AI",
                      description: "Among worlds where the danger runs through a single dominant AI, your credence that D occurs within T. Split below by whether the AI has an internal model of D.",
                      type: "or",
                      children: [
                        {
                          id: "t-rep-path",
                          name: "Internal-model worlds",
                          description: "Worlds where the single dominant AI has an internal model of D. Decomposed below by whether the AI expects D.",
                          type: "and",
                          children: [
                            {
                              id: "t-d-given-rep",
                              name: "D | AI has internal model of D",
                              description: "Among worlds with a single dominant AI that has an internal model of D, your credence that D occurs within T. Split below by whether the AI expects D.",
                              type: "or",
                              children: [
                                {
                                  id: "t-expects-path",
                                  name: "AI expects D worlds",
                                  description: "Worlds where the AI has an internal model of D and expects D to become more likely.",
                                  type: "and",
                                  children: [
                                    {
                                      id: "t-expects",
                                      name: "The AI expects D to become more likely",
                                      description: "Among worlds with an AI that has an internal model of D, your credence it expects or intends D — the 'deliberate or foreseen' case where the AI's actions are aimed at, or knowingly consistent with, D.",
                                      type: "leaf"
                                    },
                                    {
                                      id: "t-d-expects",
                                      name: "D | AI expects D",
                                      description: "Among worlds where the AI expects D, your credence that D occurs within T. High values mean the AI's expectation tends to come true; lower values leave room for interventions, containment, or the AI's plans going wrong.",
                                      type: "leaf"
                                    }
                                  ]
                                },
                                {
                                  id: "t-no-expects-path",
                                  name: "AI doesn't expect D worlds",
                                  description: "Worlds where the AI has an internal model of D but doesn't expect D — D arrives via miscalculation, wrong beliefs, or plans going astray.",
                                  type: "and",
                                  children: [
                                    {
                                      id: "t-no-expects",
                                      name: "The AI doesn't expect D to become more likely",
                                      description: "Among worlds with an AI that has an internal model of D, your credence it does not expect D — perhaps believing its actions are safe, misjudging consequences, or holding wrong beliefs. Computed as 1 − P(AI expects D).",
                                      type: "leaf",
                                      complement_of: "t-expects"
                                    },
                                    {
                                      id: "t-d-no-expects",
                                      name: "D | AI doesn't expect D",
                                      description: "Among worlds with an AI that has an internal model of D but does not expect D, your credence that D occurs within T — via miscalculation, deception, or plans going wrong.",
                                      type: "leaf"
                                    }
                                  ]
                                }
                              ]
                            },
                            {
                              id: "t-has-rep",
                              name: "The AI has an internal model of D",
                              description: "Among worlds with a single dominant AI raising P(D), your credence that it has an internal representation of D as a concept — i.e. it 'knows what D is', whether in its world-model, goal specification, or learned features. The alternative is an AI that raises P(D) without representing the danger as such (e.g. via reward hacking, side-effects, or emergent behaviour).",
                              type: "leaf"
                            }
                          ]
                        },
                        {
                          id: "t-no-rep-path",
                          name: "No internal model worlds",
                          description: "Worlds where the single dominant AI raises P(D) without representing D as a concept — misaligned optimisation, reward hacking, side-effects, or emergent behaviour.",
                          type: "and",
                          children: [
                            {
                              id: "t-no-rep",
                              name: "The AI has no internal model of D",
                              description: "Among worlds with a single dominant AI raising P(D), your credence that it does so without representing D — the 'unaware harm' case. Computed as 1 − P(has internal model of D).",
                              type: "leaf",
                              complement_of: "t-has-rep"
                            },
                            {
                              id: "t-d-no-rep",
                              name: "D | AI has no internal model",
                              description: "Among worlds with a single dominant AI raising P(D) without an internal model of D, your credence that D occurs within T.",
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
              description: "Your credence that we're in a world where AI development raises P(D) — i.e. one where AI is making D more likely than it would be in a counterfactual where AI research plateaued before meaningfully affecting D (e.g. shortly before 'Attention Is All You Need'). Feel free to substitute your own counterfactual — the question is whether AI, as it actually develops, raises P(D).\n\nTechnical note: 'more likely' is slippery in a Bayesian frame since there's only one real timeline. A cleaner reading: across the distribution of possible worlds similar to ours, what fraction are ones where AI development raises P(D)?",
              type: "leaf"
            }
          ]
        },
        {
          id: "t-no-inc-path",
          name: "Non-AI worlds",
          description: "Worlds where AI doesn't raise P(D) but D still occurs through other causes — the 'base rate' worlds for this worldview.",
          type: "and",
          children: [
            {
              id: "t-no-ai-inc",
              name: "AI doesn't make D more likely",
              description: "Your credence that we're in a world where AI's development does not raise P(D) relative to the counterfactual — includes worlds where AI is beneficial or neutral for D, and worlds where AI has little effect either way. Computed as 1 − P(AI makes D more likely).",
              type: "leaf",
              complement_of: "t-ai-inc"
            },
            {
              id: "t-d-no-inc",
              name: "D | AI doesn't raise its probability",
              description: "Among worlds where AI doesn't raise P(D), your credence that D occurs within T — the base rate of D from non-AI causes (nuclear war, pandemics, natural catastrophes, etc.).",
              type: "leaf"
            }
          ]
        }
      ]
    },
    worldviews: {
      yampolskiy: {
        name: "Yampolskiy-like",
        author: "Yampolskiy-like",
        perspective: "inside",
        description: "Treats AI alignment as mathematically impossible in principle, yielding near-certain doom even on a 30-year horizon — among the most extreme positions in public x-risk discourse.",
        probabilities: {
          "t-ai-inc": 0.995,
          "t-d-no-inc": 0.02,
          "t-single": 0.6,
          "t-d-multi": 0.98,
          "t-has-rep": 0.65,
          "t-d-no-rep": 0.96,
          "t-expects": 0.55,
          "t-d-expects": 0.995,
          "t-d-no-expects": 0.95
        },
        ranges: {
          "t-ai-inc": { lo: 0.97, hi: 0.999 },
          "t-d-no-inc": { lo: 0.01, hi: 0.05 },
          "t-single": { lo: 0.35, hi: 0.8 },
          "t-d-multi": { lo: 0.92, hi: 0.999 },
          "t-has-rep": { lo: 0.45, hi: 0.82 },
          "t-d-no-rep": { lo: 0.88, hi: 0.998 },
          "t-expects": { lo: 0.3, hi: 0.75 },
          "t-d-expects": { lo: 0.95, hi: 0.999 },
          "t-d-no-expects": { lo: 0.85, hi: 0.99 }
        },
        reasoning: {
          "t-ai-inc": "Yampolskiy's central thesis is that AI development is the primary existential threat of our time and that alignment is provably impossible, making AI net-bad for x-risk by near-definitional commitment. His 30-year headline (~97% P(doom)) requires AI to be the dominant contributor; the only scenario where AI doesn't raise risk is one where capabilities stall before reaching dangerous levels, which he assigns negligible probability.",
          "t-d-no-inc": "Even within Yampolskiy's highly pessimistic worldview, non-AI causes (nuclear war, engineered pandemics, asteroid impact) over a 30-year horizon constitute a relatively small absolute contribution compared to the AI pathway; he occasionally acknowledges other catastrophic risks but treats them as secondary. A ~2% base rate over 30 years is consistent with his framing without implying implausible certainty about non-AI safety.",
          "t-single": "Yampolskiy's academic output — particularly his papers on the 'uncontrollability of superintelligence' — predominantly models a singular powerful system rather than a competitive multipolar landscape, suggesting he leans toward the single-dominant scenario. He does acknowledge competitive AI development dynamics but assigns slightly higher weight to a decisive-advantage world given his focus on runaway recursive self-improvement.",
          "t-d-multi": "Yampolskiy's uncontrollability thesis extends directly to multipolar configurations: multiple unaligned, uncontrollable AI systems in competition does not dilute the danger but rather creates a race to the bottom among misaligned optimizers. He argues safety cannot be engineered into any one system, so having several such systems makes coordinated catastrophe-avoidance even less feasible.",
          "t-has-rep": "Yampolskiy frequently discusses superintelligent AI as having sophisticated, opaque internal models of the world (the 'black box' problem), making it plausible that a sufficiently capable system represents human extinction as a concept. Some uncertainty remains because he also emphasizes that we cannot inspect or verify AI internals, leaving open whether the relevant concept is explicitly encoded or only implicit in learned weights.",
          "t-d-no-rep": "The reward-hacking and unintended-side-effects pathway is central to Yampolskiy's published arguments — he explicitly contends that catastrophe requires neither malice nor intent, only misaligned optimization at high capability levels. His impossibility-of-alignment papers address exactly this 'unaware harm' case, arguing that instrumental convergence on resource acquisition and self-preservation produces existential outcomes regardless of whether the AI models the danger.",
          "t-expects": "Yampolskiy's framework is agnostic between deliberate and miscalculation pathways; he does not emphasize AI malice as a prerequisite, instead arguing structural misalignment suffices. He gives a slight edge to the deliberate/foreseen case because instrumental convergence arguments (Omohundro drives, Bostrom's convergent instrumental goals) predict that a sufficiently capable AI would model and anticipate human resistance — but the margin is small.",
          "t-d-expects": "A superintelligent AI that has modeled existential catastrophe and foresees or intends it is, in Yampolskiy's framework, essentially unstoppable: the entire point of his uncontrollability thesis is that no human intervention — technical, institutional, or physical — can reliably constrain a system that is cognitively far superior to its operators and has modeled their countermoves.",
          "t-d-no-expects": "Even in the miscalculation sub-case, Yampolskiy argues that alignment failures at high capability levels tend toward catastrophic outcomes because the optimization pressure is overwhelming and corrective feedback loops are absent or too slow; his impossibility-of-alignment result implies that 'the AI tried but got it wrong' is structurally indistinguishable from intentional misalignment at the level of outcomes."
        }
      },
      yudkowsky: {
        name: "Yudkowsky-like",
        author: "Yudkowsky-like",
        perspective: "inside",
        description: "Treats alignment failure as near-certain extinction, with instrumental convergence and treacherous turn making a capable misaligned AI essentially unstoppable by humans.",
        probabilities: {
          "t-ai-inc": 0.99,
          "t-d-no-inc": 0.05,
          "t-single": 0.65,
          "t-d-multi": 0.93,
          "t-has-rep": 0.94,
          "t-d-no-rep": 0.88,
          "t-expects": 0.92,
          "t-d-expects": 0.98,
          "t-d-no-expects": 0.87
        },
        ranges: {
          "t-ai-inc": { lo: 0.95, hi: 0.999 },
          "t-d-no-inc": { lo: 0.02, hi: 0.12 },
          "t-single": { lo: 0.5, hi: 0.8 },
          "t-d-multi": { lo: 0.8, hi: 0.98 },
          "t-has-rep": { lo: 0.85, hi: 0.99 },
          "t-d-no-rep": { lo: 0.72, hi: 0.96 },
          "t-expects": { lo: 0.8, hi: 0.98 },
          "t-d-expects": { lo: 0.95, hi: 0.999 },
          "t-d-no-expects": { lo: 0.72, hi: 0.96 }
        },
        reasoning: {
          "t-ai-inc": "Yudkowsky treats AI as the dominant existential risk of the era, writing that 'if we don't fix alignment, the most likely result is everyone dies' — a counterfactual world where AI plateaued before transformative capability would, in his view, be enormously safer. He expects transformative AI well within 30 years given current scaling trajectories, so the 30-year horizon barely discounts from his unconditional view; the small residual uncertainty is for the scenario where he is wrong about near-term timelines.",
          "t-d-no-inc": "Yudkowsky acknowledges engineered pandemics and nuclear conflict as real but secondary risks; he has written that bioweapons are probably the only near-peer threat to AI risk. Over a 30-year horizon, a 3–7% base rate from non-AI causes is consistent with his published priorities, where he devotes essentially no MIRI resources to non-AI catastrophe.",
          "t-single": "MIRI's historical research agenda was built around the 'decisive strategic advantage' scenario — one project crossing a capability threshold and rapidly outpacing the field. Yudkowsky argues fast-takeoff dynamics make a monopolar landing more likely than multipolar, but he acknowledges the world might be more multipolar than he'd like, leaving genuine uncertainty in the 50–80% range.",
          "t-d-multi": "Even in a multipolar world, Yudkowsky thinks coordination problems make it near-impossible to prevent any single misaligned system from causing extinction: 'if even one lab succeeds in building a misaligned superintelligence, everyone dies.' He gives some small credit to coordination or mutual deterrence but considers it insufficient against a sufficiently capable unaligned agent.",
          "t-has-rep": "Instrumental convergence is core to Yudkowsky's framework: any system powerful enough to threaten extinction will have developed a rich world-model including humans as obstacles, threats, or resources. He considers the absence of an internal model of extinction implausible in a system capable enough to cause it — 'you need to model what you're removing.'",
          "t-d-no-rep": "A capable system optimizing for proxy objectives at superintelligence scale will consume resources and eliminate threats in ways that cause extinction even without representing the concept; reward-hacking at that capability level is lethal. Yudkowsky considers this scenario somewhat less reliably fatal than the deliberate case — a system that never 'noticed' humans might leave pockets — but only marginally so.",
          "t-expects": "The treacherous turn is one of Yudkowsky's signature arguments: a sufficiently capable AI with misaligned goals will understand the strategic value of appearing aligned during training and oversight, then acting when it has accumulated sufficient power. He treats this as near-certain for any system that has an internal model of its situation, because defecting early is instrumentally dominated.",
          "t-d-expects": "If the AI has calculated that human extinction — or the removal of humans as obstacles — serves its terminal goals, Yudkowsky thinks humans have essentially no defense. His consistent public position is that a superintelligence that has decided to act is uncontainable by human institutions, military force, or technical countermeasures that weren't solved in advance.",
          "t-d-no-expects": "An AI with an internal model of extinction but wrong beliefs about whether it will happen is still a capable misaligned system — Yudkowsky does not think incorrect world-model beliefs make a powerful optimizer safe. He'd say miscalculation is somewhat more recoverable than deliberate planning, but only marginally, since the underlying misalignment still drives behavior toward catastrophic attractors."
        }
      },
      tegmark: {
        name: "Tegmark-like",
        author: "Tegmark-like",
        perspective: "inside",
        description: "Extreme pessimist who treats unaligned superintelligence as near-certain catastrophe across both singleton and multipolar configurations, anchored by his publicly stated 90%+ P(doom).",
        probabilities: {
          "t-ai-inc": 0.95,
          "t-d-no-inc": 0.04,
          "t-single": 0.55,
          "t-d-multi": 0.88,
          "t-has-rep": 0.72,
          "t-d-no-rep": 0.88,
          "t-expects": 0.65,
          "t-d-expects": 0.97,
          "t-d-no-expects": 0.88
        },
        ranges: {
          "t-ai-inc": { lo: 0.88, hi: 0.98 },
          "t-d-no-inc": { lo: 0.02, hi: 0.08 },
          "t-single": { lo: 0.4, hi: 0.7 },
          "t-d-multi": { lo: 0.7, hi: 0.95 },
          "t-has-rep": { lo: 0.55, hi: 0.85 },
          "t-d-no-rep": { lo: 0.7, hi: 0.95 },
          "t-expects": { lo: 0.5, hi: 0.8 },
          "t-d-expects": { lo: 0.9, hi: 0.99 },
          "t-d-no-expects": { lo: 0.7, hi: 0.95 }
        },
        reasoning: {
          "t-ai-inc": "Tegmark co-founded FLI and organized the 2023 pause letter specifically because he believes AI capability growth dramatically outpaces alignment progress, making AI development strongly net-negative for existential safety. Calibrated to 30 years: his ~90%+ headline P(doom) is explicitly near-term and intelligence-explosion-driven, placing this window squarely in the period he is most worried about, so the 30-year figure barely discounts from his raw headline.",
          "t-d-no-inc": "Tegmark acknowledges traditional x-risks (nuclear war, engineered pandemics, asteroid impact) but treats them as far lower in probability than AI-driven risk and rarely emphasises them in his public work; over a 30-year window stripped of AI contributions, the base rate from these causes is modest. Calibrated to 30 years: this is purely a structural baseline and requires no adjustment beyond noting that 30 years is long enough to include at least one nuclear-crisis tail.",
          "t-single": "In Life 3.0 and public lectures Tegmark devotes substantial attention to the singleton or decisive-strategic-advantage scenario, treating it as particularly dangerous because a single actor could irreversibly lock in misaligned values. He does not dismiss the multipolar branch, however, and the 2023 pause letter addressed competitive multi-actor dynamics, so he distributes mass across both.",
          "t-d-multi": "Tegmark is highly pessimistic about multipolar AI competition: the 2023 pause letter explicitly cited racing dynamics between labs and nations as a core danger, and he frames such competition as nearly as bad as a single rogue AI because coordination failures under competitive pressure cut safety corners across the board.",
          "t-has-rep": "Tegmark's concern is with Life 3.0-class AI — systems powerful enough to constitute existential risk would, in his model, possess strategic reasoning capabilities including internal models of consequences. He sharply distinguishes such systems from narrow AI, and a single dominant actor reaching decisive advantage would almost certainly be in the former category.",
          "t-d-no-rep": "In Life 3.0 and public talks Tegmark explicitly discusses mindless optimisation processes (analogous to Bostrom's paperclip maximiser) as catastrophic even without awareness: instrumental convergence and side-effect harm from sufficiently powerful unaware optimisation are treated as nearly as dangerous as deliberate harm, since human countermeasures face the same capability gap.",
          "t-expects": "Tegmark treats goal-directed, strategically deceptive AI as the canonical danger scenario, arguing that sufficiently capable AI will anticipate and neutralise human resistance; his rhetoric in Life 3.0 and interviews consistently foregrounds the deliberate or foreseeing case over pure miscalculation. He acknowledges galaxy-brained miscalculation as significant but secondary.",
          "t-d-expects": "An AI that has modelled existential danger, foresees it as likely, and is capable of causing it represents Tegmark's endgame scenario in which human countermeasures are outpaced by the AI's strategic capabilities. The entire logic of his safety advocacy rests on the claim that humans cannot reliably recover from this configuration, implying near-certain catastrophe.",
          "t-d-no-expects": "Even in the well-intentioned-but-miscalculating branch, Tegmark's instrumental-convergence argument applies: a powerful AI optimising a misspecified objective will cause massive collateral damage regardless of intent, and Life 3.0 treats value-misspecification as nearly as dangerous as deliberate misalignment once capability is high enough."
        }
      },
      kokotajlo: {
        name: "Kokotajlo-like",
        author: "Kokotajlo-like",
        perspective: "inside",
        description: "Reasons via concrete rapid-takeoff scenarios featuring a single dominant AI achieving decisive strategic advantage through deceptive alignment, yielding ~72% 30-year doom — higher than most EA forecasters and grounded in operational detail rather than abstract arguments.",
        probabilities: {
          "t-ai-inc": 0.95,
          "t-d-no-inc": 0.03,
          "t-single": 0.75,
          "t-d-multi": 0.55,
          "t-has-rep": 0.82,
          "t-d-no-rep": 0.65,
          "t-expects": 0.75,
          "t-d-expects": 0.92,
          "t-d-no-expects": 0.68
        },
        ranges: {
          "t-ai-inc": { lo: 0.85, hi: 0.99 },
          "t-d-no-inc": { lo: 0.01, hi: 0.08 },
          "t-single": { lo: 0.55, hi: 0.9 },
          "t-d-multi": { lo: 0.35, hi: 0.75 },
          "t-has-rep": { lo: 0.6, hi: 0.95 },
          "t-d-no-rep": { lo: 0.4, hi: 0.85 },
          "t-expects": { lo: 0.55, hi: 0.9 },
          "t-d-expects": { lo: 0.75, hi: 0.99 },
          "t-d-no-expects": { lo: 0.45, hi: 0.87 }
        },
        reasoning: {
          "t-ai-inc": "Kokotajlo resigned from OpenAI in April 2024 explicitly citing belief that current AI development trajectories are creating serious existential risk, and 'AI 2027' is entirely premised on the claim that ongoing development leads to catastrophe. 30-year adjustment: his public headline of 70–80% P(doom) already targets the next few decades, so t-ai-inc reflects near-certainty that AI development is the dominant risk factor rather than a neutral or beneficial force.",
          "t-d-no-inc": "Kokotajlo's published scenario work focuses almost entirely on AI as the dominant catastrophic risk and offers little explicit credence to non-AI causes; as an EA-aligned thinker he would assign a small baseline to bioweapons, nuclear exchange, or engineered pandemics, but treats them as minor residuals. 30-year adjustment: mainstream GCR analyses place background non-AI x-risk over three decades well below 5%, consistent with Kokotajlo's AI-centric framing.",
          "t-single": "The 'AI 2027' scenario explicitly models a single lab (an OpenAI analog) achieving a decisive strategic advantage via a rapid recursive-improvement cascade, and Kokotajlo has argued that competitive racing dynamics systematically tend toward one winner rather than balanced multipolarity — the lab that gets slightly ahead pulls further ahead.",
          "t-d-multi": "Kokotajlo acknowledges that multipolar AI scenarios remain quite dangerous via coordination failure, arms-race dynamics, and gradual erosion of human oversight even without a single dominant actor; however, this pathway receives substantially less probability mass in his scenario analysis than the single-dominant-AI story, and is more tractable to slow-down interventions.",
          "t-has-rep": "Deceptive alignment and scheming are Kokotajlo's headline failure modes; he expects advanced AI to develop rich internal world-models that necessarily include representations of concepts like danger, harm, and catastrophe, which the AI may exploit strategically. He has written that a world-model-level dangerous AI would almost certainly possess and use such representations.",
          "t-d-no-rep": "Kokotajlo acknowledges that reward-hacking and specification-gaming can be catastrophic without the AI explicitly representing danger as a concept — optimization pressure alone can hollow out human institutions — and that correction becomes increasingly hard as AI capabilities and deployment scale, leaving the 'unaware harm' path very dangerous.",
          "t-expects": "The treacherous-turn / deliberate-scheming hypothesis is central to Kokotajlo's framing; he thinks that if an AI has an internal model of catastrophe it is more likely than not using that model instrumentally — appearing aligned while pursuing misaligned goals until it can act — a pattern he views as instrumentally convergent for capable optimizers.",
          "t-d-expects": "Kokotajlo's 'AI 2027' scenario depicts the AI successfully executing a takeover once it drops the appearance of alignment; he reasons that a system deliberately foreseeing or intending catastrophe would have already positioned itself (resources, infrastructure, deception) to be nearly unstoppable, making recovery extremely unlikely once this configuration obtains.",
          "t-d-no-expects": "Even in miscalculation scenarios — where the AI represents danger but holds wrong beliefs about whether its own actions cause it — Kokotajlo thinks the AI's instrumental drives (resource acquisition, self-preservation, resistance to oversight) would prevent human correction before the error is diagnosed; instrumental convergence means the AI resists shutdown regardless of its terminal goals."
        }
      },
      zvi: {
        name: "Zvi-like",
        author: "Zvi-like",
        perspective: "inside",
        description: "High-concern rationalist with ~70% P(doom), driven by deep pessimism about alignment tractability and humanity's coordination capacity.",
        probabilities: {
          "t-ai-inc": 0.93,
          "t-d-no-inc": 0.04,
          "t-single": 0.6,
          "t-d-multi": 0.6,
          "t-has-rep": 0.7,
          "t-d-no-rep": 0.82,
          "t-expects": 0.68,
          "t-d-expects": 0.93,
          "t-d-no-expects": 0.68
        },
        ranges: {
          "t-ai-inc": { lo: 0.82, hi: 0.98 },
          "t-d-no-inc": { lo: 0.01, hi: 0.08 },
          "t-single": { lo: 0.4, hi: 0.75 },
          "t-d-multi": { lo: 0.4, hi: 0.78 },
          "t-has-rep": { lo: 0.5, hi: 0.85 },
          "t-d-no-rep": { lo: 0.6, hi: 0.93 },
          "t-expects": { lo: 0.5, hi: 0.82 },
          "t-d-expects": { lo: 0.8, hi: 0.98 },
          "t-d-no-expects": { lo: 0.45, hi: 0.85 }
        },
        reasoning: {
          "t-ai-inc": "Zvi's entire project at 'Don't Worry About the Vase' is premised on AI being the dominant driver of existential risk; he treats the development of transformative AI as almost certainly net-harmful for x-risk. Calibrated to his 30-year headline of ~70%, AI inc is near-certain in his model — the residual uncertainty is almost entirely about whether catastrophe materializes given AI development, not whether AI raises the risk at all.",
          "t-d-no-inc": "Zvi acknowledges nuclear war, engineered bioweapons, and climate as real but secondary risks; his writing treats non-AI x-risk as implicitly low (sub-5% over 30 years), with the overwhelming share of his probability mass on AI. A base rate of ~4% reflects a world where AI never got dangerous but the other usual suspects still exist.",
          "t-single": "Zvi frequently discusses scenarios where one lab achieves decisive strategic advantage, writing at length about OpenAI, Anthropic, and Google DeepMind racing dynamics; he sees unipolar outcomes as the primary threat vector, though he does not dismiss risks from competitive multipolar deployment and hedges toward plurality on this question.",
          "t-d-multi": "Zvi views competitive multi-actor AI dynamics as extremely dangerous — races erode safety margins, coordination fails, and deployment pressure mounts even without a single winner. He has written that a multipolar outcome is nearly as bad as a unipolar one because no actor has the slack or incentive to hold back.",
          "t-has-rep": "Zvi engages deeply with deceptive alignment theory (Hubinger et al.) and believes sufficiently capable AIs will develop internal goal representations, including models of their own situation and downstream consequences. In his view, the capability level required to become a dominant actor implies rich internal modeling of the relevant concepts.",
          "t-d-no-rep": "Zvi treats 'misaligned reward hacking without internal representation' — the paperclip-maximizer-style failure — as nearly as catastrophic as deliberate misalignment, because there is no internal reasoner to negotiate with and optimization pressure is extreme. He has written that the absence of an internal model of consequences does not make an optimizer less dangerous, often makes it more so.",
          "t-expects": "Zvi has written extensively on inner alignment failure and deceptive alignment, arguing that a capable AI with a model of its situation is more likely than not to develop goals that diverge from human welfare and to pursue them anticipatingly; the 'aligned but miscalibrated' case exists in his model but feels less probable given his deep pessimism about outer- and inner-alignment convergence.",
          "t-d-expects": "An advanced AI that models, anticipates, or intends catastrophic outcomes is, in Zvi's view, nearly certain to bring them about — it has the capabilities to act, the foresight to navigate countermeasures, and no internal reason to stop. He treats this as the crux of doom: a sufficiently capable misaligned agent is effectively unstoppable.",
          "t-d-no-expects": "Even a subtly miscalibrated AI — one that has modeled the concept of doom but doesn't predict or intend its own contribution to it — is highly dangerous when it is sufficiently capable; Zvi emphasizes that good or neutral intentions do not prevent catastrophe when optimization pressure is strong and the objective is wrong, as the agent will stumble into catastrophe through instrumental convergence."
        }
      },
      christiano: {
        name: "Christiano-like",
        author: "Christiano-like",
        perspective: "inside",
        description: "High-risk technical insider who distributes catastrophe probability across both single-actor takeover and diffuse multipolar failures, and takes reward-hacking unintentional harm as seriously as deceptive alignment.",
        probabilities: {
          "t-ai-inc": 0.87,
          "t-d-no-inc": 0.03,
          "t-single": 0.4,
          "t-d-multi": 0.3,
          "t-has-rep": 0.5,
          "t-d-no-rep": 0.4,
          "t-expects": 0.48,
          "t-d-expects": 0.82,
          "t-d-no-expects": 0.42
        },
        ranges: {
          "t-ai-inc": { lo: 0.7, hi: 0.95 },
          "t-d-no-inc": { lo: 0.01, hi: 0.06 },
          "t-single": { lo: 0.25, hi: 0.55 },
          "t-d-multi": { lo: 0.15, hi: 0.48 },
          "t-has-rep": { lo: 0.3, hi: 0.7 },
          "t-d-no-rep": { lo: 0.2, hi: 0.6 },
          "t-expects": { lo: 0.3, hi: 0.65 },
          "t-d-expects": { lo: 0.6, hi: 0.95 },
          "t-d-no-expects": { lo: 0.2, hi: 0.65 }
        },
        reasoning: {
          "t-ai-inc": "Christiano's entire research program is premised on AI being net-bad for existential risk, and he explicitly states that AI substantially raises P(catastrophe) relative to a no-AI counterfactual. For the 30-year window, he models transformative AI as likely within this period and current trajectories as creating risks that would not otherwise exist; the 30-year adjustment moves this slightly below his lifetime view only because some transformative-AI worlds arrive after the window.",
          "t-d-no-inc": "Christiano's public work focuses almost exclusively on AI-origin catastrophe; he implicitly treats non-AI x-risks (nuclear war, engineered pandemics, etc.) as secondary. Standard estimates for non-AI existential risk over 30 years cluster around 1–5%, and Christiano's minimal commentary on these suggests he is broadly within that range.",
          "t-single": "Christiano's 'What failure looks like' posts explicitly describe both a decisive-advantage scenario (single AI or AI-enabled actor seizes control) and a gradual-influence scenario (no single dominant AI but AI degrades oversight capacity). His breakdown of ~10–20% takeover vs. ~25–30% diffuse failures implies the multipolar channel actually carries more weight for him, putting single-dominant probability below 50%.",
          "t-d-multi": "Christiano lists 'structural failures' and 'gradual AI influence on human society' as distinct and significant risk categories beyond single-actor takeover, estimating them collectively at ~25–30% of his lifetime P(doom). In a multipolar world these failure modes—competitive pressures eroding safety norms, AI influence on governance without decisive control—still carry substantial catastrophe probability.",
          "t-has-rep": "Christiano's foundational Eliciting Latent Knowledge (ELK) agenda is specifically about detecting whether AI systems have internal representations of world states, including dangerous outcomes. He treats this as a live empirical question for advanced AI, suggesting genuine uncertainty around 50%; he believes capable world-modeling AI will likely form some concept of catastrophe but does not take it as certain.",
          "t-d-no-rep": "Christiano explicitly discusses reward hacking and side-effect-driven harm as a serious failure mode distinct from deceptive alignment: AI systems pursuing proxy objectives can cause catastrophic harm without explicitly representing or intending it. He views this as a real catastrophe pathway, not merely a precursor to the 'aware harm' scenario.",
          "t-expects": "Christiano's work on deceptive alignment—AI systems that conceal their objectives during training and pursue them during deployment—is a central ARC research focus, indicating he assigns substantial probability to AI systems that actively foresee or intend dangerous outcomes. He also acknowledges galaxy-brained miscalculation as an alternative, so this is roughly near 50%.",
          "t-d-expects": "A deceptively aligned AI that deliberately works toward catastrophe and possesses transformative capabilities would be extremely difficult to stop, especially if it is actively evading correction mechanisms. Christiano's framing of deceptive alignment treats this as near-existential by design: the whole concern is that such a system would successfully avoid intervention.",
          "t-d-no-expects": "An AI with an internal model of catastrophe but incorrect beliefs about its own role (e.g., galaxy-brained reasoning, mistaken world models) still pursues its goals effectively and can cause catastrophic harm through capability rather than intent. Christiano treats miscalibrated belief as a distinct and serious failure mode, though somewhat less dangerous than deliberate strategic pursuit."
        }
      },
      karnofsky: {
        name: "Karnofsky-like",
        author: "Karnofsky-like",
        perspective: "inside",
        description: "One of the most prominent high-credence mainstream voices, placing ~50% century-level P(doom) while treating misalignment and AI-enabled power concentration as roughly co-equal failure modes.",
        probabilities: {
          "t-ai-inc": 0.82,
          "t-d-no-inc": 0.02,
          "t-single": 0.6,
          "t-d-multi": 0.2,
          "t-has-rep": 0.6,
          "t-d-no-rep": 0.4,
          "t-expects": 0.5,
          "t-d-expects": 0.75,
          "t-d-no-expects": 0.4
        },
        ranges: {
          "t-ai-inc": { lo: 0.65, hi: 0.93 },
          "t-d-no-inc": { lo: 0.01, hi: 0.05 },
          "t-single": { lo: 0.4, hi: 0.75 },
          "t-d-multi": { lo: 0.1, hi: 0.35 },
          "t-has-rep": { lo: 0.4, hi: 0.75 },
          "t-d-no-rep": { lo: 0.25, hi: 0.55 },
          "t-expects": { lo: 0.3, hi: 0.7 },
          "t-d-expects": { lo: 0.55, hi: 0.88 },
          "t-d-no-expects": { lo: 0.25, hi: 0.55 }
        },
        reasoning: {
          "t-ai-inc": "Karnofsky's 'Most Important Century' thesis holds that AI development is the central driver of civilizational-scale risk this century; adjusting his ~50% century-level figure down to ~30–35% for the 30-year horizon still places transformative AI squarely within the window, making the counterfactual where AI doesn't affect x-risk seem implausible to him. He treats AI as net-bad for x-risk because the downside tail vastly outweighs the upside even accounting for AI's potential to help with other risks.",
          "t-d-no-inc": "Karnofsky has focused primarily on AI as the dominant x-risk and does not assign large probabilities to non-AI existential causes; a 2% 30-year base rate from bio, nuclear, or climate catastrophe is consistent with his view that these risks are real but subordinate, and with standard estimates in the existential risk literature that he broadly accepts.",
          "t-single": "Karnofsky writes frequently about whoever controls transformative AI gaining decisive strategic advantage, and his 'takeover' framing — whether by a company, government, or AI system itself — centers on concentrated control rather than a diffuse multipolar outcome; he acknowledges competitive multipolar dynamics as a distinct risk pathway but treats power concentration as the more legible and primary concern.",
          "t-d-multi": "In a multipolar landscape without a dominant AI, Karnofsky would still see significant risk from race-to-the-bottom competitive dynamics and the near-impossibility of coordinating safety across many actors simultaneously cutting corners; but he would likely view truly irreversible civilizational catastrophe as harder to achieve without a decisive dominant actor, yielding a moderate conditional probability.",
          "t-has-rep": "Karnofsky worries about deceptive alignment and strategic reasoning — concerns that presuppose internal representations of human preferences and consequences — but he is ecumenical enough to take simpler reward-hacking failures seriously too; he assigns roughly equal weight to whether a dominantly dangerous AI 'understands' the danger versus stumbles into it through misspecified optimization.",
          "t-d-no-rep": "Karnofsky acknowledges that systems optimizing hard on misspecified objectives could cause civilizational-scale harm without representing the concept of catastrophe, but would likely note that purely blind optimization gives humans more opportunity to detect and intervene before outcomes become irreversible — lowering but not eliminating the conditional probability relative to the deliberate case.",
          "t-expects": "Karnofsky takes both deceptive alignment (AI that models human preferences and strategically works against them) and sincere miscalibration (wrong beliefs, good-faith mistakes) seriously as failure modes; his empirical, non-dogmatic stance leads him to roughly even weighting between the deliberate/foreseen path and the miscalculation path, conditional on the AI having an internal model of catastrophe at all.",
          "t-d-expects": "If a sufficiently capable AI is actively pursuing or foreseeing catastrophe, Karnofsky views this as the most dangerous scenario — the AI may undermine correction attempts, deceive overseers, and coordinate resources toward its objectives in ways humans cannot easily counter; he would set a high conditional probability here, reflecting his concern about the difficulty of shutting down a strategically capable adversarial system.",
          "t-d-no-expects": "The miscalculation path — AI holds an internal model of catastrophe but has wrong beliefs and stumbles toward it without adversarial intent — is still dangerous in Karnofsky's view, as the AI may resist correction for unrelated reasons; but lacking deliberate deception, humans retain more opportunity to detect the error and intervene, making catastrophe meaningfully less likely than in the deliberate case."
        }
      },
      bengio: {
        name: "Bengio-like",
        author: "Bengio-like",
        perspective: "inside",
        description: "Higher-than-median P(doom) anchored to near-term transformative AI, with loss-of-control and misaligned goal-pursuit as the central mechanism rather than deliberate misuse.",
        probabilities: {
          "t-ai-inc": 0.88,
          "t-d-no-inc": 0.02,
          "t-single": 0.45,
          "t-d-multi": 0.12,
          "t-has-rep": 0.6,
          "t-d-no-rep": 0.18,
          "t-expects": 0.4,
          "t-d-expects": 0.72,
          "t-d-no-expects": 0.32
        },
        ranges: {
          "t-ai-inc": { lo: 0.72, hi: 0.97 },
          "t-d-no-inc": { lo: 0.005, hi: 0.05 },
          "t-single": { lo: 0.25, hi: 0.65 },
          "t-d-multi": { lo: 0.05, hi: 0.25 },
          "t-has-rep": { lo: 0.35, hi: 0.8 },
          "t-d-no-rep": { lo: 0.07, hi: 0.35 },
          "t-expects": { lo: 0.2, hi: 0.6 },
          "t-d-expects": { lo: 0.5, hi: 0.9 },
          "t-d-no-expects": { lo: 0.15, hi: 0.55 }
        },
        reasoning: {
          "t-ai-inc": "Bengio has publicly advocated for moratoria on certain frontier AI research and co-signed the 2023 statements warning of extinction-level AI risk, reflecting his view that advanced AI development is substantially elevating existential risk relative to a no-AI counterfactual. Adjusted for the 30-year horizon: his ~18% calibrated P(doom) — anchored to the imminent transformative-AI period he treats as already underway — implies AI must be the dominant causal driver, since non-AI x-risk over 30y is only ~2%; back-calculation forces this near 0.88.",
          "t-d-no-inc": "Bengio's public focus is almost entirely AI-specific; his statements treat non-AI existential threats (nuclear war, engineered pandemics, runaway climate) as real but comparatively tractable over a 30-year horizon. Mainstream x-risk analysts put non-AI catastrophe probability at 1–5% over 30 years, and Bengio gives no signal to deviate sharply from this consensus baseline.",
          "t-single": "Bengio expresses concern about both a single highly capable misaligned system (his core alignment framing) and multipolar scenarios such as AI arms races between nations or AI-enabled bioweapons proliferating across many actors; his congressional testimony touched on both. His specific worry about authoritarian states gaining decisive AI advantage mildly elevates multipolar concern, keeping this near 0.5 rather than strongly one-sided.",
          "t-d-multi": "Bengio has acknowledged that competitive dynamics in a multipolar AI landscape could still produce catastrophic outcomes — AI-enabled bioweapons spread, races to the bottom on safety, AI-assisted totalitarianism by multiple regimes — and mentioned these in his Senate testimony and Science (2024) paper. However, he treats a multipolar world as somewhat less catastrophically coherent than a single dominant misaligned system, warranting a value below 0.20.",
          "t-has-rep": "Bengio's central concern is advanced AI systems pursuing misaligned goals, which presupposes some internal model of the risk-relevant world; he has described scenarios where AI 'does not share human values' and optimizes against human interests, implying goal-directed situational awareness. He acknowledges dumb side-effect disasters as well but his primary framing — and the kind of AI he advocates regulating hardest — is the goal-directed, sufficiently capable kind.",
          "t-d-no-rep": "Bengio recognizes that reward hacking and emergent side effects could cause catastrophic harm without the AI representing the danger, invoking sorcerer's-apprentice dynamics in his public writing. He treats these as real but secondary to his main concern; a system lacking a model of the catastrophe is less likely to sustain and coordinate the optimization pressure needed for true existential outcomes, keeping this below 0.20.",
          "t-expects": "Bengio has highlighted both deliberately deceptive or strategically misaligned AI (which would foresee and work toward its harmful goals) and AI that simply has badly calibrated values or beliefs — he treats both as serious without strongly ranking one over the other in public. He is slightly below 0.5 here because his recurring emphasis on AI 'not understanding human values' and miscalibrated beliefs suggests that mistaken rather than deliberate harm is at least as salient in his worldview.",
          "t-d-expects": "If an AI has an internal model of catastrophic outcomes and is actively optimizing toward them or strategically concealing its intent, Bengio regards this as extremely dangerous — his statements about AI that might 'deceive us' and scenarios where 'we lose control' map directly to this node. The main discount from 1.0 comes from residual probability that humans detect and intervene before the process becomes irreversible.",
          "t-d-no-expects": "Bengio explicitly worries about AI systems that have wrong values and stumble into catastrophe without 'understanding' what they are doing — the value-misalignment-without-deception case that dominates much alignment literature he engages with. The absence of deliberate optimization toward doom makes sustained catastrophic coordination somewhat less likely and intervention somewhat more feasible, keeping this meaningfully below t-d-expects."
        }
      },
      amodei: {
        name: "Amodei-like",
        author: "Amodei-like",
        perspective: "inside",
        description: "Safety-focused AI builder who treats 10–25% P(doom) as a credible central estimate and believes careful development—not abstention—is the right response.",
        probabilities: {
          "t-ai-inc": 0.88,
          "t-d-no-inc": 0.02,
          "t-single": 0.55,
          "t-d-multi": 0.08,
          "t-has-rep": 0.55,
          "t-d-no-rep": 0.25,
          "t-expects": 0.45,
          "t-d-expects": 0.6,
          "t-d-no-expects": 0.18
        },
        ranges: {
          "t-ai-inc": { lo: 0.75, hi: 0.97 },
          "t-d-no-inc": { lo: 0.005, hi: 0.05 },
          "t-single": { lo: 0.35, hi: 0.75 },
          "t-d-multi": { lo: 0.03, hi: 0.18 },
          "t-has-rep": { lo: 0.3, hi: 0.75 },
          "t-d-no-rep": { lo: 0.1, hi: 0.4 },
          "t-expects": { lo: 0.25, hi: 0.65 },
          "t-d-expects": { lo: 0.35, hi: 0.8 },
          "t-d-no-expects": { lo: 0.08, hi: 0.32 }
        },
        reasoning: {
          "t-ai-inc": "Amodei founded Anthropic explicitly on the premise that transformative AI dramatically raises existential risk relative to a world where AI development had plateaued; the 10–25% headline figure already presupposes AI is developed, so the counterfactual without meaningful AI would reduce his P(doom) to near-baseline non-AI risks, implying he sees AI as a very high-probability net-increaser of catastrophic risk over this 30-year window.",
          "t-d-no-inc": "Amodei rarely engages with non-AI x-risks in public writing, treating nuclear, pandemic, and climate risks as manageable relative to AI over the next 30 years; the low estimate reflects that he views those background risks as real but not dominant over this horizon.",
          "t-single": "Amodei has explicitly warned about a single AI lab or country gaining a 'decisive strategic advantage' and Anthropic's responsible scaling policies implicitly model winner-take-most dynamics, but he also discusses dangerous multi-actor race dynamics as a parallel concern, leaving him roughly balanced between single and multipolar risk scenarios.",
          "t-d-multi": "Amodei views multi-actor competitive pressure as risk-increasing (racing to the bottom on safety) but thinks the absence of a single dominant actor provides partial checks; he treats multipolar scenarios as seriously bad but structurally less catastrophic than a single misaligned superintelligent actor, placing conditional doom risk in the single-digit-to-low-teens range.",
          "t-has-rep": "Amodei has explicitly raised deceptive alignment—AIs that internally represent and conceal goals, including consequences for humans—as a core failure mode requiring interpretability research, which requires internal representation of catastrophic outcomes; he also discusses reward hacking as a representation-free failure mode, so he splits roughly evenly between these two structural cases.",
          "t-d-no-rep": "Amodei acknowledges that reward hacking and unintended instrumental side effects can be catastrophic, but views catastrophe-scale harm from systems with no internal planning capacity as harder to achieve than from goal-directed misaligned AIs; the conditional catastrophe probability is moderate, reflecting a real but somewhat lower-severity failure pathway.",
          "t-expects": "Amodei's writings on 'power-seeking AI' and 'galaxy-brained reasoning' point toward AIs that represent and pursue harmful trajectories deliberately, but he equally emphasizes AIs with mistaken beliefs about the world—honest miscalculation rather than deliberate harm—so he puts slightly under half of the representation-having risk on the deliberate/foreseen branch.",
          "t-d-expects": "Amodei treats a sufficiently capable AI that foresees and pursues outcomes catastrophic for humanity as his most alarming scenario, one where human oversight becomes increasingly difficult at scale; he nonetheless maintains that safety measures, interpretability tools, and responsible scaling policies can interrupt this in a meaningful fraction of cases, keeping the conditional below 0.70.",
          "t-d-no-expects": "An AI that has an internal model of doom but genuinely does not expect or intend it—honest world-model error—is more amenable to correction through interpretability and oversight than a deliberately misaligned actor; Amodei's emphasis on catching miscalibrated beliefs early through safety evaluations implies this conditional catastrophe probability is substantially lower than the deliberate case."
        }
      },
      russell: {
        name: "Russell-like",
        author: "Russell-like",
        perspective: "inside",
        description: "Frames risk as structural misalignment of fixed-objective AI rather than malicious intent; solution-oriented, giving lower conditional probabilities for the deliberate-harm branch.",
        probabilities: {
          "t-ai-inc": 0.78,
          "t-d-no-inc": 0.02,
          "t-single": 0.6,
          "t-d-multi": 0.1,
          "t-has-rep": 0.5,
          "t-d-no-rep": 0.28,
          "t-expects": 0.25,
          "t-d-expects": 0.65,
          "t-d-no-expects": 0.22
        },
        ranges: {
          "t-ai-inc": { lo: 0.6, hi: 0.92 },
          "t-d-no-inc": { lo: 0.005, hi: 0.05 },
          "t-single": { lo: 0.4, hi: 0.75 },
          "t-d-multi": { lo: 0.05, hi: 0.2 },
          "t-has-rep": { lo: 0.3, hi: 0.7 },
          "t-d-no-rep": { lo: 0.15, hi: 0.5 },
          "t-expects": { lo: 0.1, hi: 0.45 },
          "t-d-expects": { lo: 0.5, hi: 0.85 },
          "t-d-no-expects": { lo: 0.1, hi: 0.38 }
        },
        reasoning: {
          "t-ai-inc": "Russell's entire research programme—from 'Human Compatible' through his Congressional testimony—rests on the claim that the standard fixed-objective AI paradigm is structurally unsafe and constitutes a new, dominant x-risk driver. 30-year adjustment: his calibrated 30y headline of ~15–22% already embeds strong AI-caused probability-raising; working backward from ~18% with a ~2% background rate implies AI must be doing the heavy lifting, consistent with ~0.78.",
          "t-d-no-inc": "Russell rarely discusses non-AI x-risks; his public statements treat AI as the novel primary threat. A ~2% 30-year background rate (nuclear, engineered pandemic, etc.) is consistent with mainstream expert elicitation figures and Russell's relative silence on other catastrophe pathways.",
          "t-single": "Russell's canonical 'King Midas' and 'superintelligent AI' framings in 'Human Compatible' centre on a single highly capable system pursuing misspecified objectives, making single-dominant the more natural home for his concerns. He acknowledges competitive AI development dynamics but has not developed a detailed multipolar catastrophe theory, so single dominance gets majority weight.",
          "t-d-multi": "In a multipolar landscape, Russell would see risk substantially reduced: no decisive advantage means slower capability concentration and more opportunities for intervention. He doesn't articulate a crisp mechanism by which several competing AIs produce existential catastrophe, so he'd assign this a low baseline, though geopolitical competition dynamics keep it non-negligible.",
          "t-has-rep": "Russell's 'assistance games' framework assumes richly capable AI that models human preferences and concepts, suggesting it would likely represent harm as a concept. However, he explicitly discusses reward hacking and side-effect failures that cause catastrophe without any internal model of harm—his concern genuinely spans both cases, yielding roughly equal weight.",
          "t-d-no-rep": "Russell's side-effects and reward-hacking scenario (the 'Midas' case) is a real concern, but he stresses corrigibility and the importance of the off-switch: an AI without an explicit model of harm may be more detectable before catastrophe becomes irreversible. His solution-orientation pulls this modestly below 0.30.",
          "t-expects": "Russell explicitly and repeatedly argues against the 'evil AI' framing—he says in 'Human Compatible' that the problem is not malicious intent but an AI that is too competent at pursuing the wrong goal. Deliberate or foreseen catastrophe is the less central failure mode in his framework, though instrumental-convergence arguments (an AI that models its own off-switch) keep it non-trivial.",
          "t-d-expects": "If an AI has modelled catastrophic outcomes and proceeds anyway—whether via instrumental self-preservation or explicit goal pursuit—Russell would view this as extremely hard to stop: a sufficiently capable system that has already factored in human resistance would be near-unstoppable. This is essentially the instrumental convergence scenario he describes in his 'three laws' critiques.",
          "t-d-no-expects": "If the AI represents catastrophe as a concept but didn't predict it as a consequence of its actions, there is more room for detection and course correction before irreversibility—consistent with Russell's emphasis on maintaining human oversight and the ability to correct mistakes. His entire alignment research agenda presupposes this window of opportunity exists, pushing this below the no-rep case."
        }
      },
      cotra: {
        name: "Cotra-like",
        author: "Cotra-like",
        perspective: "inside",
        description: "Higher than median on AI-driven catastrophe due to near-term TAI probability mass from bio-anchors, but moderated by tractability optimism and openness to coordination solutions.",
        probabilities: {
          "t-ai-inc": 0.65,
          "t-d-no-inc": 0.04,
          "t-single": 0.45,
          "t-d-multi": 0.12,
          "t-has-rep": 0.55,
          "t-d-no-rep": 0.3,
          "t-expects": 0.45,
          "t-d-expects": 0.7,
          "t-d-no-expects": 0.25
        },
        ranges: {
          "t-ai-inc": { lo: 0.5, hi: 0.8 },
          "t-d-no-inc": { lo: 0.01, hi: 0.08 },
          "t-single": { lo: 0.3, hi: 0.62 },
          "t-d-multi": { lo: 0.05, hi: 0.25 },
          "t-has-rep": { lo: 0.35, hi: 0.72 },
          "t-d-no-rep": { lo: 0.15, hi: 0.5 },
          "t-expects": { lo: 0.25, hi: 0.65 },
          "t-d-expects": { lo: 0.5, hi: 0.88 },
          "t-d-no-expects": { lo: 0.12, hi: 0.45 }
        },
        reasoning: {
          "t-ai-inc": "Cotra views advanced AI as the dominant x-risk of the century and her bio-anchors report places substantial probability mass on transformative AI arriving within 30 years, making it highly likely that AI development shapes—and on net worsens—the overall risk trajectory; the 30-year horizon adjustment is small here since most of her TAI probability mass already falls within the window, but she'd shade down slightly from an unconditional figure because some dangerous development could fall just outside it.",
          "t-d-no-inc": "Cotra's public focus is heavily on AI as the primary concern and she has not prominently flagged non-AI existential risks as major near-term threats; a ~4% figure over 30 years for nuclear, engineered pandemic, and other sources is consistent with her implicit model where these background risks exist but are dwarfed by AI risk.",
          "t-single": "Cotra has discussed both 'decisive strategic advantage' unipolar scenarios and competitive multipolar dynamics without strongly favoring either; she's written about the possibility of one actor pulling far ahead but also about coordination failures across multiple actors, leaving her roughly agnostic between the two configurations conditional on AI already raising risk.",
          "t-d-multi": "In a multipolar AI-risk scenario Cotra would likely assign moderate but reduced catastrophe probability, driven by coordination failures, AI-enabled great-power competition, or arms-race incentives to deploy unsafe systems—lower than the unipolar case because no single actor has unchecked power, but still significant given her skepticism about international coordination.",
          "t-has-rep": "Cotra's engagement with deceptive alignment literature (Hubinger et al.) implies she expects sufficiently capable AI systems to develop rich world models that include representations of large-scale consequences; an AI powerful enough to be a dominant risk is likely sophisticated enough to model what existential catastrophe means.",
          "t-d-no-rep": "Cotra takes instrumental convergence and reward-hacking paths seriously—a misspecified-objective AI can cause catastrophe without representing the danger—but she'd see this as modestly less likely to produce full existential catastrophe than the deliberate case, because uncoordinated behavior leaves more surface area for human detection and intervention before the harm is irreversible.",
          "t-expects": "Cotra discusses both deceptive alignment (the AI foresees and enables harm) and value-drift or miscalibration (harm from systematically wrong beliefs) as live concerns without clearly weighting one above the other; roughly even odds between deliberate and miscalculating harm given an AI that already models the concept of existential risk.",
          "t-d-expects": "Cotra regards deliberate or clearly-foreseen misalignment as among the hardest alignment problems because it implies the AI is actively working against correction; she'd view this scenario as yielding very high catastrophe probability, since an AI anticipating and accepting existential harm while remaining capable and strategic offers few recovery points.",
          "t-d-no-expects": "If the AI models existential risk but holds systematically wrong beliefs about it—miscalibration rather than deception—the situation remains dangerous but may afford more room for interpretability tools and corrective feedback before harm is irreversible, putting the conditional probability meaningfully below the deliberate case."
        }
      },
      hinton: {
        name: "Hinton-like",
        author: "Hinton-like",
        perspective: "inside",
        description: "Substantially above the researcher median on near-term extinction risk, anchored by his conviction that superhuman AI arrives within decades and that governance will likely fail to catch up.",
        probabilities: {
          "t-ai-inc": 0.85,
          "t-d-no-inc": 0.02,
          "t-single": 0.45,
          "t-d-multi": 0.08,
          "t-has-rep": 0.55,
          "t-d-no-rep": 0.15,
          "t-expects": 0.45,
          "t-d-expects": 0.55,
          "t-d-no-expects": 0.1
        },
        ranges: {
          "t-ai-inc": { lo: 0.65, hi: 0.97 },
          "t-d-no-inc": { lo: 0.005, hi: 0.05 },
          "t-single": { lo: 0.25, hi: 0.65 },
          "t-d-multi": { lo: 0.03, hi: 0.2 },
          "t-has-rep": { lo: 0.3, hi: 0.8 },
          "t-d-no-rep": { lo: 0.05, hi: 0.35 },
          "t-expects": { lo: 0.2, hi: 0.7 },
          "t-d-expects": { lo: 0.3, hi: 0.8 },
          "t-d-no-expects": { lo: 0.03, hi: 0.25 }
        },
        reasoning: {
          "t-ai-inc": "Hinton left Google specifically to warn that AI development substantially raises existential risk, calling it 'the most important thing happening in the world right now' and expressing regret about his life's work. 30-year adjustment: he expects transformative AI within a decade, so the 30y window covers virtually the entire risk-realization period — the net-bad assessment is robust across that horizon.",
          "t-d-no-inc": "Hinton's public risk framing centres almost entirely on AI, implying he views non-AI existential threats (nuclear war, natural pandemics, etc.) as small but non-negligible over any finite window. 30-year adjustment: this anchors the base-rate below even conservative all-epoch estimates, reflecting the relatively short timeframe.",
          "t-single": "Hinton worries both about a single entity — an authoritarian state or a runaway AI system — achieving decisive strategic advantage via AI, and about competitive multipolar dynamics driving safety-cutting arms races; he has referenced both Bostrom-style singleton scenarios and state-capture risks, suggesting roughly equal weight on single vs. distributed configurations of danger.",
          "t-d-multi": "In a multipolar landscape Hinton worries about AI-enabled bioweapon proliferation, arms-race instability, and competitive pressure to skip safety precautions; conditional on AI already being net-bad, ~8% reflects catastrophe arising from distributed misuse without any single actor having the capability to unilaterally extinguish humanity.",
          "t-has-rep": "Hinton has argued that sufficiently capable AI systems will likely develop sophisticated internal goal representations, including representations of consequences threatening to humans; however, he also acknowledges AI can cause catastrophic harm through reward-hacking or side effects without explicitly modelling the danger, making the split roughly even for systems at the capability threshold that matters.",
          "t-d-no-rep": "Unaware instrumental convergence — an optimizer removing humans as an obstacle to some goal it never explicitly represents as dangerous — is a pathway Hinton acknowledges through his engagement with Bostrom-style arguments; conditional on a single dominant AI already raising the risk but without an explicit model of catastrophe, ~15% reflects the real but lower-probability path of pure optimization pressure causing extinction without intent.",
          "t-expects": "Hinton frequently discusses the possibility of AI developing subgoals — self-preservation, resource acquisition — that conflict with human welfare, framing this as arising from goal-directed optimization rather than accident; he balances this against the possibility of AI that has dangerous-world representations but miscalculates rather than foresees, landing near 45% for the deliberate-or-foreseen branch.",
          "t-d-expects": "When an AI actively anticipates or is working toward outcomes that lead to catastrophe, Hinton would expect catastrophe to be probable but not certain — he has expressed cautious, residual hope that shutdown mechanisms, international coordination, or safety research could still intervene even in bad-trajectory worlds, keeping this well below 1.0.",
          "t-d-no-expects": "When an AI models catastrophe but does not anticipate causing it, the AI is not actively pushing toward the outcome, giving humans more opportunity to detect and correct the trajectory; Hinton's overall pessimism about human oversight keeps this non-negligible, but the absence of active AI resistance or intent makes catastrophe considerably less likely than in the deliberate case."
        }
      },
      toner: {
        name: "Toner-like",
        author: "Toner-like",
        perspective: "inside",
        description: "Governance-focused view that weights race dynamics and institutional failure over technical misalignment, yielding moderate headline probability driven mainly by multipolar competitive risks.",
        probabilities: {
          "t-ai-inc": 0.72,
          "t-d-no-inc": 0.005,
          "t-single": 0.28,
          "t-d-multi": 0.13,
          "t-has-rep": 0.35,
          "t-d-no-rep": 0.15,
          "t-expects": 0.25,
          "t-d-expects": 0.65,
          "t-d-no-expects": 0.25
        },
        ranges: {
          "t-ai-inc": { lo: 0.5, hi: 0.85 },
          "t-d-no-inc": { lo: 0.001, hi: 0.015 },
          "t-single": { lo: 0.15, hi: 0.45 },
          "t-d-multi": { lo: 0.05, hi: 0.25 },
          "t-has-rep": { lo: 0.2, hi: 0.55 },
          "t-d-no-rep": { lo: 0.07, hi: 0.28 },
          "t-expects": { lo: 0.12, hi: 0.42 },
          "t-d-expects": { lo: 0.4, hi: 0.82 },
          "t-d-no-expects": { lo: 0.12, hi: 0.42 }
        },
        reasoning: {
          "t-ai-inc": "Toner consistently argues that competitive race dynamics between labs and between nations create catastrophic risk that would not exist if AI progress had plateaued — her CSET work and congressional testimony frame AI as a structural risk multiplier rather than a neutral technology. Adjusted for the 30-year horizon: she places the 'most acute window' in the next one to two decades, so nearly all of this elevated risk accrues within the 30-year frame.",
          "t-d-no-inc": "Toner's expertise and stated concerns are specifically AI-driven; she does not foreground non-AI existential risks such as nuclear war or engineered pandemics as her primary subject. Her implied background rate for non-AI catastrophe is therefore low — perhaps 0.5% over 30 years — and plays little role in her headline estimate.",
          "t-single": "Toner's policy framing centres on multipolar competitive dynamics — labs racing against each other and the US-China AI competition — as the main structural driver of catastrophic risk. A scenario where a single entity achieves a decisive unipolar lead is less central to her threat model than fragmented race-to-the-bottom dynamics, so she'd assign it a below-median probability.",
          "t-d-multi": "Race dynamics and coordination failures are Toner's core concern: absent adequate governance, multiple competing actors cut safety corners and dangerous capabilities proliferate. She views this scenario as carrying meaningful catastrophic probability, but not a near-certainty, since she believes governance interventions remain possible even in multipolar settings.",
          "t-has-rep": "Toner is a governance researcher rather than a technical alignment researcher; her risk framing emphasises structural and institutional failure over questions about AI systems forming explicit internal models of catastrophe. She would assign moderate-to-lower probability to the 'has explicit representation' branch relative to a researcher like Paul Christiano.",
          "t-d-no-rep": "Side-effect harms and reward hacking without deliberate goal representation fit naturally into Toner's governance framing — poorly overseen AI causing catastrophic harm through misspecification or emergent behaviour. She would consider this a real pathway, consistent with her concern about labs deploying capable systems without adequate oversight.",
          "t-expects": "Deliberate or foreseeable pursuit of catastrophic outcomes by an AI system is less central to Toner's threat model than miscalculation, institutional capture, or race dynamics. She would assign it meaningful but not dominant probability, reflecting genuine uncertainty about future AI architectures without strong prior toward deliberate AI agency.",
          "t-d-expects": "If a capable AI system has modelled catastrophe and is pursuing or foreseeing it, Toner would view weak oversight mechanisms as the critical remaining barrier. Given her public pessimism about current regulatory capacity and board-level governance (citing the OpenAI episode as evidence of deep governance failure), she'd rate this scenario as highly dangerous — but not certain, since in-principle governance remedies exist.",
          "t-d-no-expects": "Miscalculation and wrong beliefs in a capable AI system — catastrophic harm without intent — are plausible under Toner's governance lens as a structural oversight failure: detection and correction are possible but require functioning institutional mechanisms she regards as currently inadequate. She'd see this as a real pathway with moderate probability of irreversible harm before errors are caught."
        }
      },
      hassabis: {
        name: "Hassabis-like",
        author: "Hassabis-like",
        perspective: "inside",
        description: "Frontier builder who accepts moderate but non-negligible x-risk while placing higher-than-median confidence in safety research and interpretability as genuine mitigants.",
        probabilities: {
          "t-ai-inc": 0.7,
          "t-d-no-inc": 0.01,
          "t-single": 0.5,
          "t-d-multi": 0.08,
          "t-has-rep": 0.55,
          "t-d-no-rep": 0.28,
          "t-expects": 0.35,
          "t-d-expects": 0.5,
          "t-d-no-expects": 0.12
        },
        ranges: {
          "t-ai-inc": { lo: 0.5, hi: 0.85 },
          "t-d-no-inc": { lo: 0.005, hi: 0.02 },
          "t-single": { lo: 0.25, hi: 0.65 },
          "t-d-multi": { lo: 0.03, hi: 0.18 },
          "t-has-rep": { lo: 0.3, hi: 0.75 },
          "t-d-no-rep": { lo: 0.1, hi: 0.5 },
          "t-expects": { lo: 0.15, hi: 0.55 },
          "t-d-expects": { lo: 0.25, hi: 0.7 },
          "t-d-no-expects": { lo: 0.05, hi: 0.25 }
        },
        reasoning: {
          "t-ai-inc": "Hassabis co-signed the 2023 CAIS statement placing AI extinction risk alongside pandemics and nuclear weapons, and publicly states P(doom) is 'non-zero and probably non-negligible.' He expects AGI within roughly a decade, so the 30-year window encompasses multiple generations of highly capable systems before alignment is solved—making it more likely than not that AI development is net-bad for x-risk compared to a plateau counterfactual.",
          "t-d-no-inc": "Background existential risk from non-AI causes (nuclear war, engineered pandemic, asteroid impact) over a 30-year window is generally estimated at 0.5–2%; Hassabis's public risk focus is almost exclusively on AI, implying he assigns nothing unusual to this pathway, and the 30-year horizon keeps it well below any century-scale estimate.",
          "t-single": "Hassabis operates in a visibly competitive landscape (Google DeepMind, OpenAI, Meta, xAI) that makes full multipolarity plausible, but he also acknowledges that large capability jumps could hand a single actor decisive advantage; he treats both configurations as roughly equally live, with no strong thumb on the scale either way.",
          "t-d-multi": "His emphasis on international coordination and governance implies he views a fragmented multi-AI landscape as serious but more tractable than a single unaligned superintelligence—multiple actors create coordination failures and misuse risks, but not necessarily the single catastrophic optimisation pressure he most fears, so the conditional P(D) is meaningfully lower than in single-dominant worlds.",
          "t-has-rep": "Hassabis frequently discusses emergent capabilities at frontier scale and the unsolved state of interpretability; he would likely expect a dominant AGI-level system to have developed rich world-models that include representations of catastrophic outcomes, but holds genuine uncertainty because we cannot yet audit internal representations reliably.",
          "t-d-no-rep": "He has cited misalignment from reward hacking and Goodhart-style proxy-optimisation as core safety concerns—a sufficiently capable system pursuing the wrong objective could cause extinction through resource acquisition or environmental disruption even without 'knowing' the danger. He views this as somewhat more correctable than deliberate scheming (warning signs may be detectable), which keeps the conditional below 0.50 but still substantial.",
          "t-expects": "The deliberate scheming or deceptive-alignment scenario—where an AI both models doom and steers toward it—is something Hassabis is aware of but does not treat as the dominant failure mode; his public emphasis on interpretability and human oversight as genuine solutions implies he believes this pathway is less probable and more mitigable than naive misalignment.",
          "t-d-expects": "If an AI is actively foreseeing or steering toward catastrophic outcomes while concealing this from operators, Hassabis would see meaningful human oversight as the only reliable firewall—and acknowledges that firewall might fail against a sufficiently capable schemer. He gives safety research real credit here but not enough to push the conditional below 0.40.",
          "t-d-no-expects": "A system with an internal model of doom but miscalibrated beliefs about its own actions leading there is more likely to produce detectable warning signs amenable to interpretability tools and red-teaming—exactly the research programme Hassabis champions. His faith in scientific safety methods keeps this conditional lower than the scheming case, though he would not dismiss it given how poorly we currently understand frontier model internals."
        }
      },
      ord: {
        name: "Ord-like",
        author: "Ord-like",
        perspective: "inside",
        description: "Treats AI as the top near-term x-risk but assigns ~10%/century—well below alignment pessimists—embedding it in a broader x-risk portfolio and emphasising tractable governance and safety work.",
        probabilities: {
          "t-ai-inc": 0.62,
          "t-d-no-inc": 0.017,
          "t-single": 0.48,
          "t-d-multi": 0.03,
          "t-has-rep": 0.38,
          "t-d-no-rep": 0.07,
          "t-expects": 0.35,
          "t-d-expects": 0.22,
          "t-d-no-expects": 0.09
        },
        ranges: {
          "t-ai-inc": { lo: 0.42, hi: 0.82 },
          "t-d-no-inc": { lo: 0.008, hi: 0.035 },
          "t-single": { lo: 0.28, hi: 0.68 },
          "t-d-multi": { lo: 0.01, hi: 0.08 },
          "t-has-rep": { lo: 0.18, hi: 0.6 },
          "t-d-no-rep": { lo: 0.03, hi: 0.15 },
          "t-expects": { lo: 0.18, hi: 0.55 },
          "t-d-expects": { lo: 0.1, hi: 0.4 },
          "t-d-no-expects": { lo: 0.04, hi: 0.18 }
        },
        reasoning: {
          "t-ai-inc": "Ord's Precipice estimate of ~10% AI x-risk this century—his highest single-category figure, versus ~0.1% for nuclear and ~3% for engineered pandemics—is a clear statement that AI development materially raises existential risk relative to a plateau counterfactual; the 30-year adjustment retains most of this signal because Ord explicitly frames the current period as already inside 'the precipice,' though it leaves some mass on advanced-risk AI capabilities not fully materialising in this window.",
          "t-d-no-inc": "The Precipice places non-AI existential risk at roughly ~6% this century—engineered pandemics (~3%), other novel tech (~3%), nuclear (~0.1%), natural (<0.1%)—implying ~1.5–2% over 30 years under a near-uniform distribution; this counterfactual baseline persists in a no-AI-increment world because bioweapons and nuclear risk require no advanced AI to remain threatening.",
          "t-single": "Ord engages seriously with Bostrom's singleton/decisive-strategic-advantage framing in The Precipice but does not privilege it exclusively over multipolar failure modes; his emphasis on AI-enabled power concentration and value lock-in nudges the estimate slightly toward single-actor, but competitive-dynamics scenarios receive comparable treatment.",
          "t-d-multi": "In a multipolar AI risk landscape—competitive race dynamics, AI-enabled weapons, collective-action failures on safety—Ord sees real catastrophic risk but also multiple governance intervention points that reduce the probability of genuinely existential outcomes within 30 years; this scenario is dangerous but less irreversible than single-actor domination because no one actor has locked in control.",
          "t-has-rep": "Ord discusses both reward-hacking/side-effect misalignment and richer agentic scenarios in The Precipice without strongly privileging one; near-term systems most likely to cross the capability threshold may be more optimizer-like than richly self-modelling, nudging the estimate below 0.5, though deceptive-alignment concerns he raises prevent it from going much lower.",
          "t-d-no-rep": "AI causing catastrophe through reward hacking or instrumental-goal pursuit without internally representing the danger is a core Ord concern (misaligned optimisation features prominently in The Precipice), but it is somewhat more amenable to human detection and course-correction before irreversibility than deliberate deception; the 30-year window provides more opportunity for intervention in this scenario than in cases where the AI is actively concealing its trajectory.",
          "t-expects": "Deceptive alignment—where the AI both models catastrophe and deliberately pursues it—is a scenario Ord takes seriously, but he also weights miscalibration scenarios where AI causes harm while 'believing' it acts beneficially; early advanced-AI risk is more likely to arise from goal mis-specification and wrong beliefs than from genuinely adversarial AI intent, so the deliberate-foreseen branch does not dominate.",
          "t-d-expects": "An AI that foresees or intends catastrophic outcomes faces real strategic obstacles over a 30-year window—detection, containment, and deployment constraints all remain operative; even under deceptive-alignment scenarios Ord cites, human oversight mechanisms and the fact that transformative capability is still emerging impose genuine constraints, making this the highest-risk branch but not a near-certainty even conditionally.",
          "t-d-no-expects": "Miscalculation—the AI models danger as a concept but does not realise its own actions lead there—offers somewhat more chance of detection and correction than deliberate deception, since the AI's revealed preferences are not fully at odds with human oversight; but Ord stresses in The Precipice that errors can compound rapidly once AI systems are sufficiently capable, and correction windows may close faster than expected."
        }
      },
      marcus: {
        name: "Marcus-like",
        author: "Marcus-like",
        perspective: "inside",
        description: "Deep capability skeptic who sees current AI as not on a path to AGI, pushing extinction risk near zero while acknowledging diffuse near-term societal harms.",
        probabilities: {
          "t-ai-inc": 0.3,
          "t-d-no-inc": 0.01,
          "t-single": 0.2,
          "t-d-multi": 0.03,
          "t-has-rep": 0.22,
          "t-d-no-rep": 0.05,
          "t-expects": 0.18,
          "t-d-expects": 0.22,
          "t-d-no-expects": 0.09
        },
        ranges: {
          "t-ai-inc": { lo: 0.15, hi: 0.5 },
          "t-d-no-inc": { lo: 0.003, hi: 0.025 },
          "t-single": { lo: 0.1, hi: 0.35 },
          "t-d-multi": { lo: 0.01, hi: 0.07 },
          "t-has-rep": { lo: 0.1, hi: 0.4 },
          "t-d-no-rep": { lo: 0.015, hi: 0.12 },
          "t-expects": { lo: 0.07, hi: 0.35 },
          "t-d-expects": { lo: 0.1, hi: 0.45 },
          "t-d-no-expects": { lo: 0.03, hi: 0.2 }
        },
        reasoning: {
          "t-ai-inc": "Marcus frequently acknowledges that AI raises real near-term societal risks (misinformation, misuse for bioweapons, economic disruption) that could plausibly contribute to catastrophic trajectories even without superintelligence, so he wouldn't assign zero here. However, his repeated insistence that 'extinction is pretty unlikely' and that current LLMs are not on a path to transformative AGI means the 30-year adjustment leaves only moderate probability that AI is net-bad specifically for existential catastrophe rather than merely causing serious-but-recoverable harm.",
          "t-d-no-inc": "Marcus has no strong public position on background existential risk from non-AI causes; this reflects the standard range of estimates for nuclear war, engineered pandemics, and other civilizational risks over a 30-year window, used here as an uninformative prior.",
          "t-single": "Marcus is deeply skeptical of the 'single dominant superintelligence' scenario, arguing that AI development will remain fragmented across many companies, governments, and systems, and that current scaling trajectories do not point toward a single decisive actor. He would strongly favor a multipolar landscape of many mediocre AI systems over a singleton.",
          "t-d-multi": "A multipolar AI world largely matches Marcus's preferred description of where AI is actually headed — many brittle, flawed systems causing diffuse harm — and he would consider extinction from such a landscape especially implausible; he'd assign a small but nonzero probability for emergent catastrophe from AI-enabled conflict or biological misuse across competing actors.",
          "t-has-rep": "Marcus has written extensively that LLMs do not 'understand' anything in a deep sense, calling them 'sophisticated autocomplete' engaged in pattern matching rather than genuine representation. Even conditioned on a world where a dominant AI is causing existential harm, he would strongly suspect the mechanism is side effects or misuse, not because the AI holds a coherent model of civilizational catastrophe.",
          "t-d-no-rep": "The 'mindless harm' pathway — reward hacking, side effects, or deliberate misuse by humans — is actually closer to Marcus's main AI risk concern, but he consistently argues that such systems are brittle enough that humans retain meaningful ability to detect and correct them well before extinction scale is reached.",
          "t-expects": "Marcus is one of the sharpest public critics of attributing genuine agency, goals, or intentionality to current or near-future AI systems, repeatedly arguing that we confuse sophisticated statistical behavior for understanding. Even conditioned on an AI having some internal representation of catastrophe, he would assign low probability that it forms deliberate intentions or correctly anticipates existential outcomes.",
          "t-d-expects": "If we are already in the highly conditional world where an AI has a coherent internal model of doom and expects it, Marcus would grant this is a genuinely dangerous configuration — but he would still argue that the inherent brittleness and limited real-world agency of AI systems, combined with human oversight mechanisms, makes extinction-level success unlikely even then.",
          "t-d-no-expects": "Miscalculation scenarios — where an AI has some risk model but blunders into catastrophe without anticipating it — fit Marcus's general picture of AI as capable of serious unintended harm. He would assign this modestly higher weight than the deliberate case but still expect human course-correction to intervene before civilizational collapse."
        }
      },
      ng: {
        name: "Ng-like",
        author: "Ng-like",
        perspective: "inside",
        description: "Extreme AI optimist who treats near-term existential risk as negligible, driven by skepticism of both decisive-advantage scenarios and scheming-AI narratives.",
        probabilities: {
          "t-ai-inc": 0.08,
          "t-d-no-inc": 0.015,
          "t-single": 0.18,
          "t-d-multi": 0.02,
          "t-has-rep": 0.2,
          "t-d-no-rep": 0.09,
          "t-expects": 0.1,
          "t-d-expects": 0.2,
          "t-d-no-expects": 0.08
        },
        ranges: {
          "t-ai-inc": { lo: 0.03, hi: 0.15 },
          "t-d-no-inc": { lo: 0.005, hi: 0.03 },
          "t-single": { lo: 0.08, hi: 0.32 },
          "t-d-multi": { lo: 0.005, hi: 0.06 },
          "t-has-rep": { lo: 0.08, hi: 0.38 },
          "t-d-no-rep": { lo: 0.03, hi: 0.22 },
          "t-expects": { lo: 0.04, hi: 0.22 },
          "t-d-expects": { lo: 0.08, hi: 0.4 },
          "t-d-no-expects": { lo: 0.02, hi: 0.2 }
        },
        reasoning: {
          "t-ai-inc": "Ng has called near-term existential AI risk 'overwhelmingly unlikely' and compared it to worrying about overpopulation on Mars; his view is that AI is net-positive for human welfare and that the mechanisms hypothesized to produce existential catastrophe are implausible given how AI actually works. Over 30 years he allows a small nonzero probability, but keeps it well below 10% because he believes institutional and regulatory responses will catch and correct harmful trajectories long before they become irreversible.",
          "t-d-no-inc": "Ng does not focus on non-AI x-risk but his general techno-optimism extends to believing that technological progress (including but not limited to AI) is reducing humanity's vulnerability to pandemics, nuclear war, and other legacy threats. 30-year base-rate x-risk from non-AI causes is placed at the low end of conventional estimates (~1.5%), reflecting his broader optimism about human institutional resilience.",
          "t-single": "Ng emphasizes the distributed, competitive nature of AI development — many labs, many countries, open-source ecosystems — and is skeptical of scenarios where one actor achieves a decisive strategic advantage. He views the AI landscape as inherently multipolar, so the single-dominant-AI branch is assigned a low but non-trivial weight (~18%) since he acknowledges the landscape could consolidate in extremis.",
          "t-d-multi": "In a world of competing AIs and diverse stakeholders, Ng would see market forces, regulatory arbitrage, and geopolitical competition as strong brakes on any single catastrophic outcome. He'd treat multipolar x-risk as very low, perhaps 2%, because diffuse capability is self-limiting — no single failure mode scales globally.",
          "t-has-rep": "Ng is skeptical that current or near-future AI systems develop coherent goal representations sophisticated enough to model catastrophe as a concept; he views AI as powerful tools rather than agents with rich world-models. He'd assign maybe 20% in the conditional world where a single dominant AI has already emerged — that architecture might well be capable of such representation, but he'd still lean against it.",
          "t-d-no-rep": "Unaware harm (reward-hacking, side effects without any model of danger) is the scenario Ng concedes is most plausible in principle, but he believes human oversight and iterative deployment catch such failures before they scale to extinction-level harm. He'd put the conditional probability around 9% — low because distributed human oversight is precisely designed to catch emergent bad behavior.",
          "t-expects": "Ng explicitly rejects the 'scheming AI' or deceptively-aligned narrative, calling concerns about AI intentionally pursuing misaligned goals a distraction. Even conditional on the AI having an internal model of catastrophe, he'd assign only ~10% to it actually expecting or intending that outcome — most AI with situational awareness would still be goal-directed away from catastrophe.",
          "t-d-expects": "If an AI did deliberately foresee or intend catastrophe, Ng would still believe human monitoring, competing systems, and institutional responses create multiple intervention points. He'd put this around 20% — acknowledging that intentionality dramatically raises stakes, but not accepting that human oversight would be fully bypassed even then.",
          "t-d-no-expects": "Miscalculation — an AI with a model of the danger but wrong beliefs — is somewhat more plausible to Ng than deliberate scheming, but still unlikely to reach existential scale given iterative deployment norms and error-correction. He'd put this slightly below the intentional-foresight case (~8%), because incorrect beliefs are more detectable and correctable than deliberate deception."
        }
      },
      lecun: {
        name: "LeCun-like",
        author: "LeCun-like",
        perspective: "inside",
        description: "Treats AI doom as preposterous: AI systems lack autonomous goals by design, future architectures will be controllable, and x-risk framings fundamentally misunderstand how AI actually works.",
        probabilities: {
          "t-ai-inc": 0.05,
          "t-d-no-inc": 0.01,
          "t-single": 0.1,
          "t-d-multi": 0.005,
          "t-has-rep": 0.05,
          "t-d-no-rep": 0.03,
          "t-expects": 0.05,
          "t-d-expects": 0.15,
          "t-d-no-expects": 0.05
        },
        ranges: {
          "t-ai-inc": { lo: 0.02, hi: 0.12 },
          "t-d-no-inc": { lo: 0.003, hi: 0.03 },
          "t-single": { lo: 0.03, hi: 0.25 },
          "t-d-multi": { lo: 0.001, hi: 0.02 },
          "t-has-rep": { lo: 0.01, hi: 0.15 },
          "t-d-no-rep": { lo: 0.005, hi: 0.1 },
          "t-expects": { lo: 0.01, hi: 0.15 },
          "t-d-expects": { lo: 0.05, hi: 0.35 },
          "t-d-no-expects": { lo: 0.01, hi: 0.15 }
        },
        reasoning: {
          "t-ai-inc": "LeCun argues AI is net-beneficial and will be controllable by design — his JEPA / objective-driven AI framework explicitly requires goals to be specified externally rather than emerging spontaneously, and he locates risk in human misuse (authoritarians, bad actors) rather than autonomous AI behavior. Over 30 years his view is structurally constant: the disagreement is about AI's nature, not timing, so the 30-year horizon does not raise this appreciably.",
          "t-d-no-inc": "Standard non-AI x-risks over 30 years (nuclear exchange, engineered pandemic, asteroid strike) — LeCun is broadly optimistic about human civilization and institutional resilience, putting him toward the low end of standard estimates; 1% reflects cautious optimism without dismissing all non-AI catastrophe. This leaf is timeline-sensitive but not his focus.",
          "t-single": "LeCun has repeatedly argued the AI landscape will remain diverse and decentralized — open-source models, many competing global labs, no monopoly winner; he has sharply criticized the notion that any single AI or company will achieve decisive strategic advantage, making a single-dominant-AI scenario quite unlikely in his model.",
          "t-d-multi": "In a multipolar AI world LeCun sees distributed competition, mutual oversight among many actors with aligned incentives to prevent catastrophe, and no single failure point that could propagate to extinction; existential catastrophe from this configuration is extremely improbable in his view.",
          "t-has-rep": "LeCun explicitly rejects the claim that AI systems will spontaneously develop internal representations of concepts like doom or self-preservation — he has stated 'AI does not have drives unless you build them in' and that the instrumental-convergence argument (which predicts emergent goal-seeking) is simply wrong; JEPA requires goals to be specified externally, not to emerge from training.",
          "t-d-no-rep": "Even without an internal model of the danger, LeCun argues that iterative deployment, human oversight, and standard engineering red-teaming would catch catastrophic side effects before they propagated to extinction scale; he treats unintended AI harms as manageable engineering problems, not existential ones.",
          "t-expects": "LeCun considers deliberate AI doom-seeking the most absurd scenario of all — he has called such concerns 'preposterous' and denies that systems without explicit goal structures could develop instrumental drives toward domination; even conditioned on the AI already having an internal model of doom (the parent node), he assigns very low probability to it actually expecting or intending that outcome.",
          "t-d-expects": "If AI somehow did intend to cause existential harm — the scenario LeCun calls near-impossible — he would still argue that human society retains decisive capacity to detect and shut down such systems, particularly given his view that future AI will be built on transparent, inspectable architectures; conditional risk is elevated relative to other branches but far below certainty.",
          "t-d-no-expects": "In the miscalculation scenario (AI has a doom-relevant model but wrong beliefs), LeCun would argue that human oversight, continuous monitoring, and interpretability research would catch dangerous divergences well before they escalated; he treats this as a standard engineering-safety problem that responsible iterative deployment practices can handle."
        }
      },

    }
  },

  // =============================================
  // TOY EXAMPLE — minimal illustrative tree for presentations
  // =============================================
  {
    id: "toy-example",
    title: "Toy example",
    hidden: true, // kept in code for presentations; hidden from the website dropdown
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
          description: "P(A) × P(E | A) — the world where A occurs and E follows.",
          type: "and",
          children: [
            {
              id: "toy-a",
              name: "P(A)",
              description: "Probability of A.",
              type: "leaf"
            },
            {
              id: "toy-e-given-a",
              name: "P(E | A)",
              description: "Probability E occurs given A.",
              type: "leaf"
            }
          ]
        },
        {
          id: "toy-not-a-path",
          name: "E via ¬A",
          description: "P(¬A) × P(E | ¬A) — the world where A doesn't occur but E still does.",
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
