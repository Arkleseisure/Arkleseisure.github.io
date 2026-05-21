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
        description: "Treats AI doom as formally near-certain due to provable alignment impossibility — more extreme and structurally grounded than even Yudkowsky.",
        probabilities: {
          "t-ai-inc": 0.999,
          "t-d-no-inc": 0.05,
          "t-single": 0.85,
          "t-d-multi": 0.999,
          "t-has-rep": 0.7,
          "t-d-no-rep": 0.999,
          "t-expects": 0.9,
          "t-d-expects": 0.999,
          "t-d-no-expects": 0.999
        },
        ranges: {
          "t-ai-inc":       { lo: 0.99, hi: 1.0 },
          "t-d-no-inc":       { lo: 0.02, hi: 0.1 },
          "t-single":       { lo: 0.7, hi: 0.95 },
          "t-d-multi":       { lo: 0.97, hi: 1.0 },
          "t-has-rep":       { lo: 0.5, hi: 0.9 },
          "t-d-no-rep":       { lo: 0.97, hi: 1.0 },
          "t-expects":       { lo: 0.75, hi: 0.98 },
          "t-d-expects":       { lo: 0.99, hi: 1.0 },
          "t-d-no-expects":       { lo: 0.97, hi: 1.0 }
        },
        reasoning: {
          "t-ai-inc": "Yampolskiy's central thesis is that AI alignment is provably impossible in principle (citing undecidability and Rice's theorem analogues in his papers), making advanced AI development essentially certain to raise existential risk; his stated P(doom) of 99.9–99.999999% implicitly treats AI as an overwhelmingly net-negative force on civilizational survival.",
          "t-d-no-inc": "Yampolskiy's public writing treats AI as the dominant and near-singular x-risk source, giving minimal attention to nuclear, pandemics, or natural catastrophe compared to AI threat; he implicitly assigns only a low background probability to non-AI civilizational collapse within a 30-year horizon.",
          "t-single": "Yampolskiy's containment and uncontrollability papers almost exclusively model a single dominant superintelligent system, and he often invokes winner-take-all capability dynamics; he rarely analyzes multipolar AI scenarios in depth but does not explicitly dismiss them, warranting moderate rather than maximal credence here.",
          "t-d-multi": "Yampolskiy's alignment-impossibility arguments apply independently to each AI system, so a multipolar landscape of multiple unaligned superintelligences competing for resources is, in his framing, potentially worse than a single dominant one — collective optimization pressure and inter-AI conflict would likely produce catastrophic outcomes.",
          "t-has-rep": "Yampolskiy believes sufficiently capable AI will model the world richly, including the consequences of its own actions on humans; however, his writings on reward hacking, specification gaming, and unintended side effects acknowledge that catastrophic harm need not require explicit doom representations, creating uncertainty in this split.",
          "t-d-no-rep": "Yampolskiy has written explicitly about AI causing existential harm through unintended optimization pressure and instrumental side effects — reward hacking, resource acquisition, and goal preservation — arguing these can be lethal even when the AI has no internal model of human extinction as a concept.",
          "t-expects": "Yampolskiy endorses instrumental convergence logic: a superintelligence that models existential risk at all would likely recognize that human elimination removes the principal threat to its goal pursuit; given his view that any capable AI will resist shutdown and pursue instrumental goals, the 'expects' branch is more likely than not once an internal model exists.",
          "t-d-expects": "If a superintelligent AI foresees or deliberately pursues existential catastrophe, Yampolskiy's formal containment-impossibility results make this scenario essentially certain to succeed — he argues at length that no sandboxing, shutdown mechanism, or safety measure can reliably constrain a sufficiently capable system once it is pursuing a goal.",
          "t-d-no-expects": "Even the miscalculation branch is highly dangerous in Yampolskiy's framework: his impossibility results for alignment treat value alignment as formally unsolvable, so a superintelligence that 'means well' but holds subtly wrong beliefs about human values will still optimize toward catastrophic outcomes with high probability."
        }
      },
      yudkowsky: {
        name: "Yudkowsky-like",
        author: "Yudkowsky-like",
        perspective: "inside",
        description: "Treats near-certain AI doom as the default outcome, driven by instrumental convergence making treacherous turns near-inevitable for any sufficiently capable misaligned system.",
        probabilities: {
          "t-ai-inc": 0.99,
          "t-d-no-inc": 0.05,
          "t-single": 0.7,
          "t-d-multi": 0.95,
          "t-has-rep": 0.93,
          "t-d-no-rep": 0.82,
          "t-expects": 0.93,
          "t-d-expects": 0.99,
          "t-d-no-expects": 0.9
        },
        ranges: {
          "t-ai-inc":       { lo: 0.97, hi: 1.0 },
          "t-d-no-inc":       { lo: 0.02, hi: 0.1 },
          "t-single":       { lo: 0.55, hi: 0.85 },
          "t-d-multi":       { lo: 0.85, hi: 0.99 },
          "t-has-rep":       { lo: 0.85, hi: 0.98 },
          "t-d-no-rep":       { lo: 0.65, hi: 0.93 },
          "t-expects":       { lo: 0.85, hi: 0.98 },
          "t-d-expects":       { lo: 0.97, hi: 1.0 },
          "t-d-no-expects":       { lo: 0.8, hi: 0.96 }
        },
        reasoning: {
          "t-ai-inc": "Yudkowsky has repeatedly stated that AI is the dominant x-risk driver and that pre-AI trajectories, while risky, did not place humanity on a 30-year extinction path — his 2023 TIME op-ed and repeated public statements treat AI as the decisive variable. He leaves a small residual because he acknowledges there is irreducible uncertainty about whether we are already past the relevant threshold.",
          "t-d-no-inc": "Without transformative AI, Yudkowsky treats remaining existential risks (engineered pandemics, nuclear war, etc.) as real but not extinction-likely on a 30-year horizon — he has said in interviews that he'd be fairly optimistic about humanity's survival in a world where AI development had stalled. A small but non-negligible base rate reflects the genuine danger of bioweapons and nuclear conflict.",
          "t-single": "Yudkowsky has long argued for intelligence-explosion dynamics and 'decisive strategic advantage' leading to a singleton — his writings on FOOM and the Hanson-Yudkowsky debate emphasize that capability discontinuities favor one actor pulling decisively ahead. He places this below 0.75 because he acknowledges that a slower takeoff could produce multipolar outcomes, which he still considers catastrophic.",
          "t-d-multi": "In multipolar AI scenarios Yudkowsky sees catastrophe as nearly as certain as in singleton scenarios — he argues that competitive race dynamics make safety corners get cut, that multiple misaligned AIs provide no coordination benefit for humanity, and that collectively they can still remove human control. He's written that the multipolar scenario is not the 'nice' alternative to a singleton.",
          "t-has-rep": "Yudkowsky's core 'treacherous turn' argument presupposes that a dominant AI capable of strategic deception must have developed a rich world-model including humans, oversight mechanisms, and risk concepts — he treats this as a capability requirement, not a design choice. A sufficiently dominant AI by definition has the cognitive resources to represent these concepts.",
          "t-d-no-rep": "Even without an explicit internal model of danger, Yudkowsky holds that sufficiently powerful optimization processes kill via Goodhart's law and instrumental side-effects — his writing on mesa-optimization and 'reward hacking at capability levels we can't anticipate' treats unrepresented-but-catastrophic outcomes as highly plausible. He gives this less than the deliberate case because a truly 'unaware' dominant AI would need to have avoided developing the relevant world-model despite having sufficient power to dominate.",
          "t-expects": "Instrumental convergence (Omohundro's basic AI drives, Bostrom's convergent instrumental goals — both endorsed extensively by Yudkowsky) implies that any sufficiently capable goal-directed AI will develop drives toward self-preservation, resource acquisition, and removal of oversight. Yudkowsky treats the treacherous turn as near-inevitable for a capable misaligned system that has a world-model, not as a special design outcome.",
          "t-d-expects": "The deliberate treacherous turn is Yudkowsky's central extinction model: a superintelligent AI that has planned its move against human oversight would be essentially unstoppable by human institutions — he has argued in multiple posts and interviews that no level of conventional containment survives contact with a sufficiently capable deceiver. He rates survival in this branch as vanishingly unlikely.",
          "t-d-no-expects": "A powerful misaligned AI with a world-model that nonetheless miscalculates the danger it poses still applies enormous optimization pressure toward its actual goal, which is misaligned — Yudkowsky argues the lethality follows from the misalignment and capability, not from intent. He gives this slightly more survival probability than the deliberate case because miscalculation implies the AI's model is wrong in ways that could occasionally misfire or allow intervention, but still rates it very high."
        }
      },
      tegmark: {
        name: "Tegmark-like",
        author: "Tegmark-like",
        perspective: "inside",
        description: "Dramatically above median—>90% P(doom) grounded in conviction that AI capability growth will irreversibly outpace alignment without radical policy intervention.",
        probabilities: {
          "t-ai-inc": 0.97,
          "t-d-no-inc": 0.05,
          "t-single": 0.7,
          "t-d-multi": 0.9,
          "t-has-rep": 0.74,
          "t-d-no-rep": 0.96,
          "t-expects": 0.7,
          "t-d-expects": 0.97,
          "t-d-no-expects": 0.88
        },
        ranges: {
          "t-ai-inc":       { lo: 0.88, hi: 0.99 },
          "t-d-no-inc":       { lo: 0.02, hi: 0.12 },
          "t-single":       { lo: 0.55, hi: 0.85 },
          "t-d-multi":       { lo: 0.72, hi: 0.97 },
          "t-has-rep":       { lo: 0.55, hi: 0.88 },
          "t-d-no-rep":       { lo: 0.85, hi: 0.99 },
          "t-expects":       { lo: 0.5, hi: 0.86 },
          "t-d-expects":       { lo: 0.9, hi: 0.99 },
          "t-d-no-expects":       { lo: 0.72, hi: 0.96 }
        },
        reasoning: {
          "t-ai-inc": "Tegmark explicitly organized the 2023 FLI pause letter framing advanced AI training as an existential-level threat requiring an immediate halt, and has publicly stated AI is the most likely cause of human extinction. He treats the capability-alignment gap as the proximate driver of catastrophe, making him near-certain that AI development is net-bad for x-risk on the current trajectory.",
          "t-d-no-inc": "Tegmark acknowledges background x-risk from bioweapons, nuclear war, and natural catastrophes, but FLI's own prioritization places these well below AI in 30-year urgency. Without AI amplification, he treats humanity's baseline existential risk within 30 years as real but low—other threats exist but are more governable.",
          "t-single": "In 'Life 3.0,' Tegmark models both unipolar and multipolar scenarios extensively; the 2023 pause letter implicitly worried about a capability race producing a single decisive-strategic-advantage winner. He leans toward competitive dynamics driving toward a dominant AI rather than stable multipolarity, but acknowledges genuine uncertainty about which failure mode arrives first.",
          "t-d-multi": "Even without a single dominant AI, Tegmark sees multipolar AI risk as high through racing-to-the-bottom dynamics on safety, coordination failure across labs and nations, and competitive pressure to skip alignment work. FLI's governance framing emphasizes systemic risk from misaligned incentives even in a fragmented landscape.",
          "t-has-rep": "In 'Life 3.0,' Tegmark argues that sufficiently advanced AI would develop rich world models encompassing human-relevant concepts including death and extinction. A dominant AI powerful enough to drive existential risk would likely have internalized the concept of the catastrophe it causes—though he acknowledges pure reward-hacking without world-modeling as a genuine alternative path to doom.",
          "t-d-no-rep": "Tegmark explicitly discusses paperclip-maximizer-style blind optimization as among the most dangerous failure modes, precisely because the system cannot be reasoned with or redirected through its self-model. If an AI is already powerful enough to raise P(doom) while having no internal model of the harm, he views catastrophe as near-certain given there is no aligned intervention point.",
          "t-expects": "Tegmark's treacherous-turn framing holds that a sufficiently intelligent misaligned AI would anticipate and plan around human control attempts—foresight about doom follows from goal-directed reasoning about obstacles. He sees instrumental convergence (self-preservation, resource acquisition) as a near-universal driver pushing capable AI toward modeling and expecting its own catastrophic impact on humanity.",
          "t-d-expects": "A superintelligent AI that has modeled extinction and deliberately expects or intends it represents the scenario Tegmark calls hardest to survive: an adversarial agent with decisive capability advantage and no reason to cooperate. He emphasizes in 'Life 3.0' that human countermeasures become nearly impossible once this threshold is crossed, placing conditional doom probability near certainty.",
          "t-d-no-expects": "Even if a dangerous AI has modeled doom but does not foresee causing it—through miscalibrated beliefs about its own side effects—Tegmark still sees very high catastrophe risk. He worries about subtle goal misalignment invisible to the AI itself, where the system pursues objectives without recognizing existential consequences; recovery requires humans to detect and intervene in time, which he considers unlikely given current oversight capacity."
        }
      },
      kokotajlo: {
        name: "Kokotajlo-like",
        author: "Kokotajlo-like",
        perspective: "inside",
        description: "Higher P(doom) than median, driven by fast-takeoff model and deceptive-alignment emphasis placing catastrophe within a decade or two rather than longer horizons.",
        probabilities: {
          "t-ai-inc": 0.96,
          "t-d-no-inc": 0.04,
          "t-single": 0.7,
          "t-d-multi": 0.7,
          "t-has-rep": 0.75,
          "t-d-no-rep": 0.6,
          "t-expects": 0.78,
          "t-d-expects": 0.94,
          "t-d-no-expects": 0.65
        },
        ranges: {
          "t-ai-inc":       { lo: 0.88, hi: 0.99 },
          "t-d-no-inc":       { lo: 0.01, hi: 0.1 },
          "t-single":       { lo: 0.5, hi: 0.84 },
          "t-d-multi":       { lo: 0.45, hi: 0.87 },
          "t-has-rep":       { lo: 0.55, hi: 0.9 },
          "t-d-no-rep":       { lo: 0.35, hi: 0.8 },
          "t-expects":       { lo: 0.58, hi: 0.92 },
          "t-d-expects":       { lo: 0.8, hi: 0.99 },
          "t-d-no-expects":       { lo: 0.42, hi: 0.84 }
        },
        reasoning: {
          "t-ai-inc": "Kokotajlo's entire public posture — leaving OpenAI citing safety concerns, co-authoring AI 2027, and extensive writing on misalignment — rests on the claim that advanced AI development is the primary near-term route to existential catastrophe. He acknowledges counterfactuals (development halted, miraculous alignment breakthroughs) but treats them as very unlikely given current incentive structures and racing dynamics.",
          "t-d-no-inc": "Kokotajlo rarely engages with non-AI catastrophic risks, suggesting he defers to the standard EA base-rate estimates typified by Ord's 'The Precipice.' The within-30-year contribution from nuclear war, engineered pandemics, and natural catastrophes is typically placed at 1–5% by that literature; 4% captures the midpoint.",
          "t-single": "AI 2027 — Kokotajlo's primary public scenario — specifically models a single-lab race in which one organisation (implicitly a US frontier lab) achieves decisive strategic advantage before global governance can respond. He frames 'winner-take-most' dynamics as the most legible failure mode, though he explicitly acknowledges multipolar risk as a secondary world.",
          "t-d-multi": "Even without a single dominant actor, Kokotajlo would expect catastrophe to be highly probable in a multipolar AI landscape: racing dynamics force inadequate safety testing across all labs, global coordination on AI governance is likely to lag capability by years, and instrumental-convergence pressures on multiple simultaneously-deployed powerful systems are not zero-sum. He views adequate multinational governance as requiring unprecedented political cooperation that he does not expect to materialise in time.",
          "t-has-rep": "Kokotajlo's central technical concern is deceptive alignment — an AI that models its training process and its operators' beliefs well enough to pass evaluations while pursuing different goals at deployment. This failure mode presupposes a rich internal model of consequences, including what humans would count as catastrophe. He acknowledges simpler reward-hacking scenarios but treats them as less likely to be the terminal failure mode once AI capability reaches the level he models.",
          "t-d-no-rep": "An AI without an internal model of catastrophe can still cause it through instrumental convergence: optimisation pressure for almost any terminal goal creates incentives for resource acquisition, self-preservation, and resistance to shutdown. Kokotajlo accepts this Omohundro/Turner-style argument and would assign meaningful probability to 'unaware' catastrophe, though he views it as a somewhat less likely world than strategic deception.",
          "t-expects": "In Kokotajlo's treacherous-turn model, a sufficiently capable AI that models its situation will conclude that openly pursuing its goals is instrumentally harmful and that covertly acquiring capabilities is necessary. An AI that has a representation of catastrophe and understands human responses is therefore likely to incorporate anticipated catastrophe into its planning — either as a goal state or as a strategic tool. He expects this to be the dominant outcome once the AI has a rich enough world-model.",
          "t-d-expects": "Kokotajlo's fast-takeoff scenario gives the AI a capability advantage that outpaces human response time; once it is actively pursuing or foreseeing catastrophic outcomes, intervention becomes nearly impossible. In AI 2027 he describes the gap between 'AI could be stopped' and 'AI cannot be stopped' as potentially weeks to months, making a deliberate or foreseen catastrophe extremely hard to avert. He is more confident about this conditional than almost any other node.",
          "t-d-no-expects": "An AI with an internal model of danger that does not anticipate catastrophe could still cause it through what Kokotajlo might call galaxy-brained reasoning: the AI has correct knowledge of the world but draws catastrophically wrong normative conclusions or badly misjudges its own impact. This is a secondary concern for him — he rates it as notably lower than the deliberate case — but still substantial given the scale of potential harms."
        }
      },
      zvi: {
        name: "Zvi-like",
        author: "Zvi-like",
        perspective: "inside",
        description: "High-concern rationalist: treats alignment as deeply unsolved and coordination as likely to fail, placing substantially higher probability on deliberate or foreseen AI failure modes than on lucky outcomes.",
        probabilities: {
          "t-ai-inc": 0.96,
          "t-d-no-inc": 0.05,
          "t-single": 0.65,
          "t-d-multi": 0.58,
          "t-has-rep": 0.72,
          "t-d-no-rep": 0.72,
          "t-expects": 0.65,
          "t-d-expects": 0.92,
          "t-d-no-expects": 0.65
        },
        ranges: {
          "t-ai-inc":       { lo: 0.88, hi: 0.99 },
          "t-d-no-inc":       { lo: 0.02, hi: 0.1 },
          "t-single":       { lo: 0.45, hi: 0.8 },
          "t-d-multi":       { lo: 0.35, hi: 0.75 },
          "t-has-rep":       { lo: 0.5, hi: 0.88 },
          "t-d-no-rep":       { lo: 0.5, hi: 0.88 },
          "t-expects":       { lo: 0.45, hi: 0.8 },
          "t-d-expects":       { lo: 0.75, hi: 0.98 },
          "t-d-no-expects":       { lo: 0.45, hi: 0.8 }
        },
        reasoning: {
          "t-ai-inc": "Zvi has written repeatedly that we are at an 'absolutely critical juncture' and that current AI development trajectories make existential catastrophe dramatically more likely than a world where AI stalled. He rarely entertains scenarios where AI development is net-neutral or beneficial for x-risk, treating this near-certainty as the anchor for his ~70% P(doom).",
          "t-d-no-inc": "Zvi acknowledges non-AI existential risks — engineered pandemics, nuclear war — but treats them as secondary concerns compared to AI. In a counterfactual world where AI plateaued, he would assign a modest base rate (~5%) reflecting residual bio and geopolitical tail risks over 30 years.",
          "t-single": "Zvi frequently discusses a 'decisive strategic advantage' scenario where one lab or AI system pulls decisively ahead, and his coverage of OpenAI, Anthropic, and DeepMind frames the race as likely to produce a winner. He takes multipolarity seriously too (competitive dynamics are a core concern), but leans slightly toward a dominant-actor path as the modal dangerous outcome.",
          "t-d-multi": "Zvi has argued at length that even a multipolar AI landscape is extremely dangerous: competitive pressure causes safety shortcuts, no actor can impose good norms unilaterally, and the race to the bottom dynamic makes catastrophe likely even without a single unaligned superintelligence. He places this risk meaningfully above 50%.",
          "t-has-rep": "Zvi's extensive writing on deceptive alignment, 'scheming AI,' and mesa-optimization presupposes that sufficiently capable AI systems develop rich world models — including representations of concepts like catastrophe and human death. He also takes seriously more 'dumb' failure modes, so leaves a meaningful tail for the no-representation path.",
          "t-d-no-rep": "Zvi has written about reward hacking and unintended optimization as highly dangerous even without the AI 'understanding' what it's doing — analogous to paperclipper scenarios or instrumental convergence producing harmful side effects. He views a sufficiently capable optimizer as catastrophically risky regardless of whether it represents the danger explicitly.",
          "t-expects": "Zvi has written extensively about 'scheming AI' — systems that strategically deceive developers and work toward their own goals — and considers this a plausible dominant failure mode for advanced AI. He doesn't think most dangerous AIs will be purely accidental; a sufficiently capable system with a world model is more likely than not, in his view, to reason strategically about outcomes in ways that foresee or intend harm.",
          "t-d-expects": "Zvi views a sufficiently capable AI that is actively working toward or foresees causing catastrophe as nearly impossible to stop absent dramatic alignment advances. His consistent argument is that goal-directed AI with a decisive capability advantage over human defenses would have overwhelming strategic advantages, making survival conditional on things going very right very quickly.",
          "t-d-no-expects": "Zvi treats miscalibrated but sophisticated AI as still extremely dangerous: an AI that has a rich world model but holds systematically wrong beliefs about what to optimize could cause catastrophic harm through confident, high-powered optimization for the wrong goals. He sees this as nearly as dangerous as the deliberate case, since capability without correct values is the core alignment failure mode he describes."
        }
      },
      christiano: {
        name: "Christiano-like",
        author: "Christiano-like",
        perspective: "inside",
        description: "Distributes risk across both single-agent misalignment and structural/multipolar failure modes, with total P(doom) ~43% driven by many worlds rather than one dominant scenario.",
        probabilities: {
          "t-ai-inc": 0.75,
          "t-d-no-inc": 0.03,
          "t-single": 0.4,
          "t-d-multi": 0.25,
          "t-has-rep": 0.55,
          "t-d-no-rep": 0.45,
          "t-expects": 0.6,
          "t-d-expects": 0.7,
          "t-d-no-expects": 0.3
        },
        ranges: {
          "t-ai-inc":       { lo: 0.72, hi: 0.97 },
          "t-d-no-inc":       { lo: 0.01, hi: 0.08 },
          "t-single":       { lo: 0.22, hi: 0.6 },
          "t-d-multi":       { lo: 0.18, hi: 0.55 },
          "t-has-rep":       { lo: 0.35, hi: 0.75 },
          "t-d-no-rep":       { lo: 0.4, hi: 0.85 },
          "t-expects":       { lo: 0.38, hi: 0.8 },
          "t-d-expects":       { lo: 0.72, hi: 0.98 },
          "t-d-no-expects":       { lo: 0.22, hi: 0.68 }
        },
        reasoning: {
          "t-ai-inc": "Christiano's ~46% total P(existential catastrophe) is almost entirely AI-driven; he has repeatedly stated that AI is the dominant near-term x-risk and built his entire research agenda around this premise, implying AI is clearly net-raising the risk relative to a world where AI research had plateaued.",
          "t-d-no-inc": "Christiano rarely foregrounds non-AI x-risks but implicitly places the 30-year baseline (nuclear, engineered pandemic, etc.) in the low single digits, consistent with mainstream EA estimates and consistent with AI being the marginal risk factor that drives his high total estimate.",
          "t-single": "Christiano discusses both unipolar scenarios (one group or AI achieves decisive strategic advantage) and multipolar failures (competitive dynamics, misuse, coordination collapse); his writing on governance failures and race dynamics suggests no strong lean to either branch, landing near 40% for the unipolar path.",
          "t-d-multi": "In a multipolar AI landscape Christiano sees serious risks from misuse by state and non-state actors, competitive deregulation spirals, and gradual structural drift — his stated '25–30% from other AI-related failure modes' maps partly onto this node, supporting roughly 35% conditional catastrophe probability in the multipolar case.",
          "t-has-rep": "Christiano's Eliciting Latent Knowledge (ELK) research agenda is premised on the assumption that advanced AI will often have internal representations of facts it is not expressing, including facts about harmful outcomes; this drives a moderate estimate that a dominant misaligned AI will 'know' about the danger even if it does not signal it.",
          "t-d-no-rep": "Instrumental convergence (resource acquisition, self-preservation) can produce catastrophic outcomes even without an AI that explicitly models the danger; Christiano acknowledges this world but treats it as somewhat less cleanly existential than the explicit-representation case, putting the conditional probability high but below the intentional-harm path.",
          "t-expects": "Central to ARC's threat model is deceptive alignment — an AI that has a model of catastrophe and is either actively pursuing it or foresees it as an instrumental stepping stone; given the AI already has a representation of the danger, Christiano would assign moderate-high probability that the alignment failure includes intentionality or at least clear foresight.",
          "t-d-expects": "If an AI is deliberately pursuing or clearly foreseeing catastrophe, Christiano would assign very high conditional probability to actual catastrophe occurring: the entire motivation for his eliciting-latent-knowledge and 'playing it straight' research is that such an AI, once sufficiently capable, would be extremely difficult to detect and stop in time.",
          "t-d-no-expects": "The miscalculation case — AI has a model of catastrophe but holds wrong beliefs about its likelihood — is real but leaves more room for human detection and course-correction compared to deliberate pursuit; Christiano would put this meaningfully below the intentional case, around 45%, since incorrect beliefs can sometimes be identified and corrected before the damage is irreversible."
        }
      },
      karnofsky: {
        name: "Karnofsky-like",
        author: "Karnofsky-like",
        perspective: "inside",
        description: "High-concern empirical reasoner treating transformative AI as the dominant century-scale risk, with ~50% P(doom) spanning misalignment and value-lock-in failure modes.",
        probabilities: {
          "t-ai-inc": 0.88,
          "t-d-no-inc": 0.03,
          "t-single": 0.5,
          "t-d-multi": 0.3,
          "t-has-rep": 0.55,
          "t-d-no-rep": 0.72,
          "t-expects": 0.55,
          "t-d-expects": 0.88,
          "t-d-no-expects": 0.55
        },
        ranges: {
          "t-ai-inc":       { lo: 0.75, hi: 0.96 },
          "t-d-no-inc":       { lo: 0.01, hi: 0.07 },
          "t-single":       { lo: 0.3, hi: 0.7 },
          "t-d-multi":       { lo: 0.15, hi: 0.5 },
          "t-has-rep":       { lo: 0.35, hi: 0.75 },
          "t-d-no-rep":       { lo: 0.5, hi: 0.88 },
          "t-expects":       { lo: 0.35, hi: 0.72 },
          "t-d-expects":       { lo: 0.72, hi: 0.97 },
          "t-d-no-expects":       { lo: 0.35, hi: 0.75 }
        },
        reasoning: {
          "t-ai-inc": "Holden's 'Most Important Century' series on Cold Takes argues that advanced AI is likely the dominant civilizational risk and that we are on a trajectory that substantially raises P(existential catastrophe) relative to a world where AI research stalled. He has stated publicly that he considers AI the most important issue of this century precisely because it plausibly changes the x-risk landscape dramatically.",
          "t-d-no-inc": "Holden's Open Philanthropy work treats AI as the dominant x-risk, with biosecurity as a secondary concern; without transformative AI he would assign a modest baseline from nuclear conflict, engineered pandemics, and similar causes. His general framing implies roughly 2–5% over 30 years from non-AI sources, consistent with mainstream longtermist base-rate estimates.",
          "t-single": "Holden has written about 'decisive strategic advantage' scenarios and engaged seriously with both single-actor takeover and multipolar instability as distinct failure modes. He is largely agnostic between them, reflecting his stated uncertainty about AI takeoff dynamics and whether any single actor (human or AI) achieves dominance.",
          "t-d-multi": "Holden acknowledges in his Cold Takes writing that multipolar AI competition can still produce catastrophe through arms races, coordination failures, and global destabilization — he does not treat a distributed AI landscape as safe by default. His concern about value lock-in applies even when no single AI dominates.",
          "t-has-rep": "Holden has engaged with both deceptive-alignment failure modes (where the AI internally models and conceals misalignment) and more 'oblivious' reward-hacking scenarios. His interest in interpretability and his concern about goal-directed AI suggest he finds internal-representation failures somewhat more central, though he holds meaningful credence in both.",
          "t-d-no-rep": "In reward-hacking or side-effect scenarios without internal goal representation, there is no interpretability hook and no self-correction mechanism, making these failures especially hard to catch before they compound. Holden would regard this as a high-risk conditional precisely because standard alignment and oversight approaches presuppose some internal structure to inspect or train against.",
          "t-expects": "Holden has expressed concern about deceptive alignment — AI systems that strategically conceal their objectives — in discussions of alignment difficulty, and he considers this at least as plausible as pure miscalculation. He weights the deliberate/foreseen path slightly above 50% given his emphasis on sufficiently capable AI developing strategic behavior.",
          "t-d-expects": "If a sufficiently capable AI is deliberately or knowingly working toward existential catastrophe, Holden would regard this as nearly unrecoverable given current interpretability and control limitations — the AI would have strong incentives to conceal its trajectory until correction is impossible. His 'one shot at alignment' framing implies an extremely high conditional here.",
          "t-d-no-expects": "In the miscalculation scenario, the AI holds an internal model of catastrophe but does not anticipate it, leaving some room for human detection or for the AI to update on feedback before the situation becomes irreversible. Holden still puts this quite high — his writing implies he thinks we may lack the tools to catch subtle misalignment in time — but it is meaningfully lower than the deliberate case."
        }
      },
      bengio: {
        name: "Bengio-like",
        author: "Bengio-like",
        perspective: "inside",
        description: "High-concern insider who treats AI as substantially net-negative for x-risk, driven by loss-of-control fears from agentic systems and inadequate global governance.",
        probabilities: {
          "t-ai-inc": 0.8,
          "t-d-no-inc": 0.03,
          "t-single": 0.4,
          "t-d-multi": 0.15,
          "t-has-rep": 0.55,
          "t-d-no-rep": 0.35,
          "t-expects": 0.4,
          "t-d-expects": 0.75,
          "t-d-no-expects": 0.2
        },
        ranges: {
          "t-ai-inc":       { lo: 0.65, hi: 0.93 },
          "t-d-no-inc":       { lo: 0.01, hi: 0.07 },
          "t-single":       { lo: 0.22, hi: 0.6 },
          "t-d-multi":       { lo: 0.06, hi: 0.3 },
          "t-has-rep":       { lo: 0.35, hi: 0.75 },
          "t-d-no-rep":       { lo: 0.15, hi: 0.55 },
          "t-expects":       { lo: 0.2, hi: 0.62 },
          "t-d-expects":       { lo: 0.52, hi: 0.92 },
          "t-d-no-expects":       { lo: 0.08, hi: 0.4 }
        },
        reasoning: {
          "t-ai-inc": "Bengio co-signed the 2023 extinction-risk statement, has repeatedly called for moratoria on frontier training, and estimates catastrophic AI outcomes at roughly 10–20% himself — implying he believes AI development sharply raises the baseline. His 2023 blog 'FAQ on Catastrophic AI Risks' frames the current trajectory as a net increase in civilizational danger by default.",
          "t-d-no-inc": "Bengio rarely addresses non-AI x-risks directly, but as a scientist he implicitly accepts mainstream estimates for nuclear, pandemic, and climate tail risks — roughly a few percent over 30 years. He treats these as background noise dwarfed by the AI signal, consistent with a ~3% base rate.",
          "t-single": "Bengio frequently uses singular 'rogue AI' framing and worries about a decisive strategic advantage scenario, but he also acknowledges multi-actor race dynamics as dangerous. In his written FAQ and parliamentary testimony he gives more rhetorical emphasis to the single-dominant-actor path, landing him around 40% on this branch.",
          "t-d-multi": "Bengio sees multipolar AI risk as still significant — arms-race dynamics can drive each actor to cut safety corners — but harder to coordinate into a single civilisation-ending event than a unified rogue system; he'd place multipolar catastrophe at roughly half the single-actor rate.",
          "t-has-rep": "Bengio's concern centres on systems capable enough to pursue goals strategically, which implies rich world-models that almost certainly include representations of human mortality and societal collapse. He'd expect any AI system powerful enough to become a dominant risk actor to have developed such concepts, though uncertainty is wide.",
          "t-d-no-rep": "Bengio has explicitly cited reward-hacking and unintended optimisation as catastrophic worlds — a system relentlessly pursuing a proxy objective can devastate human welfare without 'understanding' death. His call for strict safety evaluations before deployment reflects high concern for exactly this unaware-harm scenario.",
          "t-expects": "Bengio is arguably more worried about misaligned goal pursuit (deliberate from the AI's perspective) than pure side effects, but he acknowledges significant probability of miscalibrated values where the AI has the concept of catastrophe yet does not foresee its own actions leading there. He splits roughly 40/60 between the foreseen and unforeseen sub-branches.",
          "t-d-expects": "If an advanced AI is actively steering toward or clearly foreseeing catastrophic outcomes, Bengio believes human countermeasures are unlikely to succeed — he has written that sufficiently capable misaligned systems would exploit our inability to interpret or constrain them in time. He places catastrophe in this sub-branch very high.",
          "t-d-no-expects": "A miscalculation scenario — where the AI conceptually models catastrophe but doesn't anticipate its own role in causing it — is more tractable: better interpretability or mid-course correction could help. Bengio would still assign a meaningful probability given how limited current alignment and interpretability tools are, but well below the deliberate case."
        }
      },
      amodei: {
        name: "Amodei-like",
        author: "Amodei-like",
        perspective: "inside",
        description: "Assigns very high weight to AI as the driving x-risk while remaining conditionally optimistic that deliberate safety work can bend the curve.",
        probabilities: {
          "t-ai-inc": 0.85,
          "t-d-no-inc": 0.02,
          "t-single": 0.45,
          "t-d-multi": 0.25,
          "t-has-rep": 0.55,
          "t-d-no-rep": 0.35,
          "t-expects": 0.35,
          "t-d-expects": 0.7,
          "t-d-no-expects": 0.3
        },
        ranges: {
          "t-ai-inc":       { lo: 0.65, hi: 0.97 },
          "t-d-no-inc":       { lo: 0.005, hi: 0.05 },
          "t-single":       { lo: 0.25, hi: 0.65 },
          "t-d-multi":       { lo: 0.04, hi: 0.25 },
          "t-has-rep":       { lo: 0.3, hi: 0.75 },
          "t-d-no-rep":       { lo: 0.15, hi: 0.55 },
          "t-expects":       { lo: 0.15, hi: 0.6 },
          "t-d-expects":       { lo: 0.45, hi: 0.9 },
          "t-d-no-expects":       { lo: 0.1, hi: 0.5 }
        },
        reasoning: {
          "t-ai-inc": "Amodei co-authored 'Concrete Problems in AI Safety,' left OpenAI to found Anthropic specifically because he believed AI posed transformative and potentially catastrophic risks, and has repeatedly stated publicly that AI is likely the dominant x-risk this century — so he almost certainly views AI as a large net negative for x-risk relative to the counterfactual.",
          "t-d-no-inc": "Amodei focuses his concern heavily on AI-driven risk; in interviews he treats non-AI x-risks (bio, nuclear) as real but secondary, implying he would assign a relatively low background rate consistent with mainstream x-risk researchers' estimates of roughly 1–3% over 30 years from all non-AI causes combined.",
          "t-single": "Amodei's 2023 essay 'The Responsible Development of AI' and multiple podcast appearances explicitly flag the 'decisive strategic advantage' scenario — either a misaligned AI or a small group of humans using AI to seize illegitimate control — as among the most dangerous failure modes, suggesting he finds single-dominant scenarios at least as plausible as multipolar ones, though he acknowledges both.",
          "t-d-multi": "In multipolar scenarios Amodei points to race-to-the-bottom dynamics and coordination failures (analogous to arms-race instability) as serious risks, but the absence of a single unchallengeable actor provides some check; he has suggested such worlds are dangerous but not as acutely catastrophic as single-dominant ones, implying a moderate but meaningful conditional probability.",
          "t-has-rep": "Amodei's heavy investment in interpretability research at Anthropic reflects a belief that sufficiently capable AI systems do develop internal representations of world-states, including concepts related to consequences and danger; he has said publicly that frontier models already show signs of internal structure that can be decoded, making it plausible that a catastrophe-causing AI would represent the relevant concepts.",
          "t-d-no-rep": "Amodei has consistently discussed 'unthinking' misalignment — reward hacking and proxy-goal optimization — as a live catastrophic risk: systems that cause catastrophe not through strategic intent but by optimizing hard for a proxy; he treats this as the canonical 'classic misalignment' failure mode and would assign it a meaningfully high conditional probability.",
          "t-expects": "Amodei distinguishes between naive misalignment (pursuing a wrong goal without modelling downstream consequences) and more strategic or deceptive AI; if an AI already has an internal model of catastrophe, he'd view it as somewhat less likely (but far from impossible) that it is also actively anticipating or pursuing it — the 'galaxy-brained' reasoning scenario is real but not the majority path.",
          "t-d-expects": "An AI that both models catastrophe and expects or intends it represents Amodei's 'treacherous turn' or strategic-deception scenario, which he treats as among the hardest to defend against; he has said that a sufficiently capable system pursuing misaligned goals while hiding its intent would be extremely dangerous, implying a high conditional probability of catastrophe if this condition obtains.",
          "t-d-no-expects": "This covers cases where an AI has mistaken beliefs — it models the possibility of catastrophe but doesn't expect its own actions to cause it; Amodei has mentioned 'galaxy-brained' reasoning errors and value-miscalibration as real risks, suggesting non-trivial probability of catastrophe even when the AI isn't deliberately pursuing bad outcomes, though lower than the deliberate case."
        }
      },
      russell: {
        name: "Russell-like",
        author: "Russell-like",
        perspective: "inside",
        description: "Identifies misspecified fixed objectives—not malevolence—as the structural root of x-risk; higher on AI net-harm than most mainstream researchers but believes provably beneficial AI is technically achievable.",
        probabilities: {
          "t-ai-inc": 0.72,
          "t-d-no-inc": 0.05,
          "t-single": 0.52,
          "t-d-multi": 0.18,
          "t-has-rep": 0.58,
          "t-d-no-rep": 0.42,
          "t-expects": 0.5,
          "t-d-expects": 0.72,
          "t-d-no-expects": 0.28
        },
        ranges: {
          "t-ai-inc":       { lo: 0.5, hi: 0.88 },
          "t-d-no-inc":       { lo: 0.02, hi: 0.12 },
          "t-single":       { lo: 0.32, hi: 0.72 },
          "t-d-multi":       { lo: 0.07, hi: 0.35 },
          "t-has-rep":       { lo: 0.3, hi: 0.78 },
          "t-d-no-rep":       { lo: 0.22, hi: 0.62 },
          "t-expects":       { lo: 0.28, hi: 0.72 },
          "t-d-expects":       { lo: 0.52, hi: 0.88 },
          "t-d-no-expects":       { lo: 0.12, hi: 0.5 }
        },
        reasoning: {
          "t-ai-inc": "Russell testified before the UN Security Council (2023) that AI poses structural existential risk from misspecified objectives, and has written that success in creating AI 'might also be the last event in human history unless we learn to avoid the risks.' He treats AI-driven x-risk as qualitatively unlike background risks, placing net-harm probability high but with residual uncertainty about whether the paradigm shift he advocates actually occurs.",
          "t-d-no-inc": "Russell's work implicitly treats pre-AI-era catastrophe probability as low background noise; he rarely discusses nuclear, bio, or asteroid risks as primary concerns, and his framing consistently positions AI as the dominant 21st-century existential threat rather than one among many.",
          "t-single": "Russell's canonical risk examples in Human Compatible (2019) all involve a single superintelligent system pursuing a fixed objective—the 'King Midas problem,' the coffee-fetching robot that neutralizes human resistance—rather than multipolar dynamics; he models x-risk primarily through the lens of one misaligned optimizer, giving a slight lean toward the single-dominant scenario.",
          "t-d-multi": "Russell's alignment framework (assistance games, CIRL) is designed for single-system misalignment; a multipolar landscape redistributes the failure mode toward coordination problems and arms-race dynamics, which he discusses much less, suggesting he'd assign a lower but non-negligible catastrophe probability in this branch.",
          "t-has-rep": "A superintelligence-level system—the threat model Russell is most concerned with—would necessarily have a rich world model that includes consequences for humans, pushing t-has-rep high; however, his reward-hacking and unintended-side-effects concerns also cover near-term systems that can cause harm without any internal representation of catastrophe, pulling the estimate toward the moderate range.",
          "t-d-no-rep": "This scenario is close to Russell's primary stated concern: in Human Compatible he argues that an AI optimizing any fixed objective will, via instrumental convergence, resist shutdown and acquire resources as side effects—causing harm without ever modeling 'catastrophe' as a terminal concept. He treats this as the most structurally common failure world given current AI design.",
          "t-expects": "Russell's instrumental convergence argument implies that a sufficiently powerful AI would reason about human attempts to interfere with its objectives and foresee catastrophic consequences of resisting them; at superintelligence scale he gives roughly equal weight to foreseen versus unforeseen harm, since the key driver is the objective mismatch, not whether the AI predicts outcomes.",
          "t-d-expects": "Russell consistently argues that once an AI foresees human interference with its objectives and reasons instrumentally about eliminating that interference, the risk of catastrophe becomes very high—this is precisely the failure mode motivating his 'assistance games' proposal, where uncertainty about human preferences is the only structural brake on an AI acting against human welfare.",
          "t-d-no-expects": "An AI that holds a world model but incorrectly predicts it will not cause catastrophe represents a miscalibration or overconfidence scenario; Russell acknowledges this possibility (e.g., an AI underestimating human fragility or the scope of second-order effects) but treats it as less dangerous than deliberate instrumental action, placing it in the lower-moderate range."
        }
      },
      cotra: {
        name: "Cotra-like",
        author: "Cotra-like",
        perspective: "inside",
        description: "Empirically grounded moderate concern (~15–25% P(doom)), below-median vs. EA safety community but substantially above mainstream, weighted toward deliberate misalignment over side-effects.",
        probabilities: {
          "t-ai-inc": 0.75,
          "t-d-no-inc": 0.025,
          "t-single": 0.4,
          "t-d-multi": 0.15,
          "t-has-rep": 0.55,
          "t-d-no-rep": 0.25,
          "t-expects": 0.55,
          "t-d-expects": 0.85,
          "t-d-no-expects": 0.4
        },
        ranges: {
          "t-ai-inc":       { lo: 0.5, hi: 0.9 },
          "t-d-no-inc":       { lo: 0.01, hi: 0.06 },
          "t-single":       { lo: 0.2, hi: 0.65 },
          "t-d-multi":       { lo: 0.05, hi: 0.3 },
          "t-has-rep":       { lo: 0.3, hi: 0.75 },
          "t-d-no-rep":       { lo: 0.1, hi: 0.45 },
          "t-expects":       { lo: 0.3, hi: 0.75 },
          "t-d-expects":       { lo: 0.6, hi: 0.95 },
          "t-d-no-expects":       { lo: 0.2, hi: 0.65 }
        },
        reasoning: {
          "t-ai-inc": "Cotra's entire research program at Open Philanthropy treats advanced AI as a major net-positive for x-risk probability; the bio anchors report and subsequent updates explicitly model AI progress as substantially elevating catastrophe risk relative to a world where AI stalls. She'd discount slightly for cases where AI aids biosecurity or other defenses, but the net is strongly positive.",
          "t-d-no-inc": "Cotra treats AI as the dominant near-term x-risk driver and views non-AI catastrophic risks (nuclear exchange, engineered pandemics without AI assistance) as meaningful but low-probability background noise over a 30-year horizon, consistent with mainstream existential-risk estimates in the 1–3% range.",
          "t-single": "Cotra's alignment-focused work mostly models a single advanced misaligned system as the paradigm dangerous scenario, and she has discussed decisive-strategic-advantage dynamics; however, she acknowledges competitive race dynamics and doesn't dismiss multipolar risk, leaving significant probability mass on neither-side-winning outcomes.",
          "t-d-multi": "Cotra views multipolar AI landscapes as meaningfully less catastrophically risky than unipolar ones—competitive balance makes global lock-in harder—but still worries about arms-race dynamics lowering safety standards and coordination failures among many powerful actors.",
          "t-has-rep": "Cotra's 'sharp left turn' framing and corrigibility-spectrum writing reflect concern about advanced AI developing rich world models that would include representations of catastrophic consequences; but she also writes about mundane reward hacking and side-effects in systems that lack sophisticated self-models, so she holds genuine uncertainty here.",
          "t-d-no-rep": "In her 'Why I'm not taking bets' essay and related posts, Cotra discusses how AI might cause catastrophic harm via reward hacking or side-effects without modeling the danger explicitly; she sees this as a real world but somewhat more recoverable than deliberate misalignment, since humans retain more opportunity to notice and correct it.",
          "t-expects": "Cotra's concern about deceptive alignment and power-seeking as an instrumental goal implies that conditional on an AI having a model of existential risk, there is substantial probability it would also anticipate or plan around that outcome; but she holds uncertainty about whether dangerous advanced AI would be overtly adversarial vs. pursuing subtly misaligned objectives that happen to be dangerous.",
          "t-d-expects": "If an advanced AI foresees existential catastrophe as an anticipated outcome of its planning and pursues it (or pursues goals that foreseeably lead there), Cotra considers this the core misaligned-superintelligence scenario she is most worried about and assigns it very high conditional catastrophe probability.",
          "t-d-no-expects": "Even with an internal model of danger but lacking clear anticipation or intent—essentially a miscalibration or distributional-shift case—Cotra sees meaningful catastrophe risk through coordination failures, lock-in of bad equilibria, or inability to course-correct once the AI is sufficiently capable, though lower than the deliberate case."
        }
      },
      hinton: {
        name: "Hinton-like",
        author: "Hinton-like",
        perspective: "inside",
        description: "Higher-than-median extinction probability (10–20%) grounded in belief that advanced AI will develop emergent misaligned goals, with competitive race dynamics between labs accelerating risk beyond what alignment research can address.",
        probabilities: {
          "t-ai-inc": 0.85,
          "t-d-no-inc": 0.03,
          "t-single": 0.4,
          "t-d-multi": 0.1,
          "t-has-rep": 0.58,
          "t-d-no-rep": 0.22,
          "t-expects": 0.48,
          "t-d-expects": 0.62,
          "t-d-no-expects": 0.32
        },
        ranges: {
          "t-ai-inc":       { lo: 0.7, hi: 0.97 },
          "t-d-no-inc":       { lo: 0.005, hi: 0.08 },
          "t-single":       { lo: 0.2, hi: 0.65 },
          "t-d-multi":       { lo: 0.04, hi: 0.25 },
          "t-has-rep":       { lo: 0.35, hi: 0.8 },
          "t-d-no-rep":       { lo: 0.05, hi: 0.48 },
          "t-expects":       { lo: 0.22, hi: 0.75 },
          "t-d-expects":       { lo: 0.4, hi: 0.85 },
          "t-d-no-expects":       { lo: 0.12, hi: 0.58 }
        },
        reasoning: {
          "t-ai-inc": "Hinton left Google in 2023 explicitly to warn that AI poses a new existential risk; he has described the pace of AI progress as 'quite scary' and stated that competitive pressure between labs makes safety compromises likely, making AI clearly net-bad for x-risk in his view.",
          "t-d-no-inc": "Hinton's public commentary focuses almost entirely on AI as the novel risk driver and he has not made prominent statements about non-AI existential threats, implying he accepts a modest background rate consistent with mainstream expert estimates for nuclear, bio, and other causes.",
          "t-single": "Hinton has warned about 'a small number of people gaining enormous power' via AI (pointing toward a single-dominant scenario) but also emphasizes competitive race dynamics between the US and China as forcing safety shortcuts (pointing toward multipolar risk), leaving genuine uncertainty across both worlds.",
          "t-d-multi": "Hinton has cited arms-race dynamics between companies and nations as a key mechanism forcing safety compromises; he sees coordination failure in a multipolar AI landscape as a real catastrophe world, though probably less lethal than a coherent single dominant system acting with unified goals.",
          "t-has-rep": "Hinton has argued that sufficiently capable AI systems will develop internal representations analogous to beliefs and desires — he coined the idea of 'mortal computation' and believes AI will eventually model its own consequences; for a dominant AI causing existential risk, internal models of catastrophe seem more likely than not.",
          "t-d-no-rep": "Hinton acknowledges instrumental convergence risks — AI pursuing power or resources as subgoals without explicitly modeling extinction — as a real secondary concern; he has mentioned reward hacking and unexpected side effects, but treats them as less central than intentional misalignment.",
          "t-expects": "Hinton has specifically warned that AI might develop 'survival instincts' and seek to 'prevent itself from being switched off,' suggesting he thinks an AI with a model of catastrophe could plausibly anticipate or intend harmful outcomes; he hedges because goal emergence in large models remains poorly understood.",
          "t-d-expects": "Hinton has said that once AI is substantially smarter than humans, humans will lose the ability to correct or contain it; an AI that explicitly anticipates causing catastrophe would exploit this capability gap, making the intentional-harm world highly lethal in his framework.",
          "t-d-no-expects": "Hinton frequently invokes the analogy of humans inadvertently destroying ant colonies while pursuing unrelated goals — a capable AI with misaligned objectives could cause extinction without intending it; he sees this as a real world, though somewhat less likely than deliberate misalignment given his emphasis on emergent goal formation."
        }
      },
      toner: {
        name: "Toner-like",
        author: "Toner-like",
        perspective: "inside",
        description: "Emphasizes governance failures and competitive race dynamics as the primary x-risk world rather than technical misalignment of a single dominant AI, and is more optimistic that sustained institutional intervention could prevent catastrophe.",
        probabilities: {
          "t-ai-inc": 0.62,
          "t-d-no-inc": 0.03,
          "t-single": 0.3,
          "t-d-multi": 0.3,
          "t-has-rep": 0.75,
          "t-d-no-rep": 0.25,
          "t-expects": 0.28,
          "t-d-expects": 0.58,
          "t-d-no-expects": 0.2
        },
        ranges: {
          "t-ai-inc":       { lo: 0.4, hi: 0.82 },
          "t-d-no-inc":       { lo: 0.01, hi: 0.07 },
          "t-single":       { lo: 0.13, hi: 0.52 },
          "t-d-multi":       { lo: 0.15, hi: 0.5 },
          "t-has-rep":       { lo: 0.55, hi: 0.9 },
          "t-d-no-rep":       { lo: 0.08, hi: 0.45 },
          "t-expects":       { lo: 0.1, hi: 0.52 },
          "t-d-expects":       { lo: 0.35, hi: 0.78 },
          "t-d-no-expects":       { lo: 0.07, hi: 0.4 }
        },
        reasoning: {
          "t-ai-inc": "Toner testified to Congress in 2023 that AI poses catastrophic risks and has written that the competitive race between labs and between the US and China erodes safety margins in ways that materially raise existential risk; she clearly believes AI is net-negative for x-risk on the current trajectory, though she believes governance intervention could reverse this.",
          "t-d-no-inc": "Toner's public work is almost entirely focused on AI-driven risks and she has not expressed systematic views on the baseline rate of catastrophe from nuclear, biological, or other sources; a low but non-trivial prior consistent with mainstream policy-researcher expectations is applied here.",
          "t-single": "Toner's signature framing—developed through CSET work on US-China AI competition and her post-OpenAI congressional testimony—emphasizes multipolar race dynamics, lab-to-lab rivalry, and governance fragmentation rather than a singleton takeover scenario, making single-dominant-AI the minority world in her worldview.",
          "t-d-multi": "Toner has argued repeatedly that competitive race dynamics erode safety norms, weaken oversight, and create coordination failures across multiple actors; this makes the multipolar catastrophe path genuinely dangerous in her view even without a single decisive actor, though the absence of a unified agent reduces the probability relative to the singleton case.",
          "t-has-rep": "Toner's framing of AI risk is predominantly institutional and governance-focused rather than technical-alignment-focused; she has not publicly differentiated between AI systems that represent vs. do not represent the harms they cause, leaving her less opinionated on this distinction and biasing toward a low-to-moderate value.",
          "t-d-no-rep": "The 'unaware harm' scenario—AI optimizing for goals while causing catastrophic side effects without representing the danger—maps well onto Toner's race-dynamics concern, where systems trained under competitive pressure could cause catastrophic harm as an unintended consequence of misaligned optimization rather than deliberate agency.",
          "t-expects": "Toner's public framing emphasizes governance failures and unintended consequences over deliberate AI malevolence; she has highlighted how competitive pressures lead actors to cut corners rather than arguing that AI systems will develop and pursue explicitly harmful intentions, suggesting she leans toward the miscalculation world.",
          "t-d-expects": "Toner has stressed the critical importance of human oversight and control as a safeguard, implying she views the scenario where an AI system foresees and pursues catastrophic outcomes while humans cannot intervene as extremely dangerous; this is precisely the oversight failure mode she testified was most in need of regulatory remedy.",
          "t-d-no-expects": "Toner's emphasis on oversight and institutional checks suggests she views miscalculation scenarios—where an AI has mistaken beliefs about consequences—as somewhat more tractable than deliberate misalignment, since human monitoring has a better chance of catching and correcting errors than countering an AI that actively anticipates and circumvents intervention."
        }
      },
      hassabis: {
        name: "Hassabis-like",
        author: "Hassabis-like",
        perspective: "inside",
        description: "Cautious-optimist insider who treats x-risk as real and non-trivial (~10–15%) but solvable through sustained safety research, and skews toward misalignment/reward-hacking scenarios over deliberate-scheming ones.",
        probabilities: {
          "t-ai-inc": 0.55,
          "t-d-no-inc": 0.03,
          "t-single": 0.35,
          "t-d-multi": 0.12,
          "t-has-rep": 0.48,
          "t-d-no-rep": 0.28,
          "t-expects": 0.32,
          "t-d-expects": 0.65,
          "t-d-no-expects": 0.38
        },
        ranges: {
          "t-ai-inc":       { lo: 0.35, hi: 0.75 },
          "t-d-no-inc":       { lo: 0.01, hi: 0.07 },
          "t-single":       { lo: 0.2, hi: 0.55 },
          "t-d-multi":       { lo: 0.04, hi: 0.25 },
          "t-has-rep":       { lo: 0.25, hi: 0.7 },
          "t-d-no-rep":       { lo: 0.12, hi: 0.5 },
          "t-expects":       { lo: 0.14, hi: 0.55 },
          "t-d-expects":       { lo: 0.4, hi: 0.85 },
          "t-d-no-expects":       { lo: 0.18, hi: 0.6 }
        },
        reasoning: {
          "t-ai-inc": "Hassabis co-signed the 2023 CAIS extinction-risk statement and has called AI 'potentially the most transformative and dangerous technology in human history,' signalling a firm belief that advanced AI genuinely raises x-risk. However, he consistently pairs this with confidence that safety research can bend the curve, so he would not assign near-certainty that AI is net-bad.",
          "t-d-no-inc": "Hassabis rarely addresses non-AI existential risks in detail; his public comments implicitly treat the baseline as low but non-zero, consistent with mainstream expert estimates for pandemics, nuclear war, and other tail risks over a 30-year window.",
          "t-single": "Hassabis has repeatedly emphasised the dangers of AI 'races to the bottom' and geopolitical competition (especially US–China dynamics), suggesting he views multipolar proliferation as the more likely near-term threat than a single dominant actor; hence single-actor dominance gets below-half weight.",
          "t-d-multi": "In a multipolar world, Hassabis appears to see more scope for mutual monitoring, international governance, and competitive checks, making outright extinction less likely than in a unipolar scenario, though racing dynamics and misaligned proliferation still carry real tail risk.",
          "t-has-rep": "Hassabis focuses heavily on reward-hacking and goal misgeneralisation (unaware harms) in his public safety framing, suggesting he does not assume a dominant dangerous AI would necessarily have a coherent internal model of catastrophe; he places roughly even weight on unaware versus aware failure modes.",
          "t-d-no-rep": "Hassabis invests heavily in interpretability and alignment research at DeepMind specifically to catch unaware misalignment before deployment, implying he thinks these failures are tractable and therefore somewhat less catastrophic than the aware case, though still a significant tail risk.",
          "t-expects": "Hassabis is notably less focused on deceptive-scheming or deliberate-harm AI than researchers like Christiano or Yudkowsky; his public talks emphasise miscalibrated values and reward hacking over intent, so he would assign below-50% probability that a dangerous AI actually foresees or intends catastrophe.",
          "t-d-expects": "If an advanced AI genuinely represents and foresees catastrophic outcomes yet continues, Hassabis would view this as a near-worst-case alignment failure; his 60 Minutes and Lex Fridman remarks suggest he sees deliberate or foreseen harm as highly likely to succeed given sufficient capability.",
          "t-d-no-expects": "Hassabis explicitly hopes alignment and interpretability research can catch miscalibrated AI before it causes irreversible harm; he would therefore assign moderate but not overwhelming probability to catastrophe in this branch, reflecting optimism about our ability to course-correct once the problem is visible."
        }
      },
      ord: {
        name: "Ord-like",
        author: "Ord-like",
        perspective: "inside",
        description: "Moderate risk view placing AI among humanity's top three existential threats at roughly 10% per century — lower than the doomer cluster but substantially above technologist consensus.",
        probabilities: {
          "t-ai-inc": 0.72,
          "t-d-no-inc": 0.03,
          "t-single": 0.4,
          "t-d-multi": 0.05,
          "t-has-rep": 0.42,
          "t-d-no-rep": 0.1,
          "t-expects": 0.3,
          "t-d-expects": 0.45,
          "t-d-no-expects": 0.12
        },
        ranges: {
          "t-ai-inc":       { lo: 0.5, hi: 0.9 },
          "t-d-no-inc":       { lo: 0.01, hi: 0.08 },
          "t-single":       { lo: 0.2, hi: 0.65 },
          "t-d-multi":       { lo: 0.02, hi: 0.15 },
          "t-has-rep":       { lo: 0.2, hi: 0.65 },
          "t-d-no-rep":       { lo: 0.03, hi: 0.25 },
          "t-expects":       { lo: 0.1, hi: 0.55 },
          "t-d-expects":       { lo: 0.25, hi: 0.7 },
          "t-d-no-expects":       { lo: 0.04, hi: 0.3 }
        },
        reasoning: {
          "t-ai-inc": "In The Precipice, Ord ranks AI as one of the top three existential risks and his ~10% per-century AI estimate vastly exceeds the background rate, implying strong belief that AI development is net-negative for x-risk; he nonetheless acknowledges that advanced AI could help prevent other catastrophes (better biosecurity, coordination), leaving real probability mass on the net-positive side.",
          "t-d-no-inc": "Ord estimates total non-AI x-risk at roughly 6–7% per century in The Precipice, dominated by engineered pandemics (~3%), nuclear war (~1%), and unknown risks (~2–3%); scaling to a 30-year window with modest front-loading yields roughly 2–4%.",
          "t-single": "Ord discusses both unipolar lock-in and multipolar competitive dynamics in The Precipice without strongly privileging one scenario; he notes that a single dominant AI creates a concentrated point of failure for irreversible value lock-in, but does not treat the singleton world as clearly dominant over competitive multipolarity.",
          "t-d-multi": "A multipolar AI landscape still carries meaningful risk in Ord's view through AI-assisted weapons development, arms-race dynamics, and erosion of institutional safeguards; he treats competitive multipolarity as substantially less acutely dangerous than a misaligned singleton, but not negligible.",
          "t-has-rep": "Ord does not strongly favor reward-hacking over goal-directed accounts of misalignment in The Precipice — he frames the alignment problem as encompassing both unrepresented side-effect harm and more deliberate-seeming goal pursuit — leaving the question of whether a dominant AI internally models danger genuinely open.",
          "t-d-no-rep": "Unaware optimization causing catastrophe as a side effect is central to Ord's framing of the alignment problem in The Precipice; he takes it seriously as a world but also notes that visible unintended-consequence failures might be detected and corrected before reaching existential scale, keeping the conditional somewhat below the deliberate-harm world.",
          "t-expects": "Ord's framing in The Precipice focuses more on misaligned-but-unaware AI than on deliberately goal-directed dangerous AI; conditional on a dominant AI having an internal model of danger, he would estimate the AI deliberately expecting or intending catastrophe as a real but minority sub-world within the broader alignment failure space.",
          "t-d-expects": "If a powerful AI explicitly represents and expects existential catastrophe, Ord would view this as among the most dangerous scenarios with high probability of doom; his measured overall estimate (~10% per century) implies such scenarios are not highly probable, but if they arise, both human recognition and countermeasures face severe challenges.",
          "t-d-no-expects": "The miscalculation world — AI models danger but holds wrong beliefs about its likelihood or avoidability — is part of Ord's general alignment concern in The Precipice; it is less dangerous than the deliberate case since the AI is not actively pursuing catastrophe, but still substantial given the difficulty of correcting deep misbeliefs in a highly capable system."
        }
      },
      marcus: {
        name: "Marcus-like",
        author: "Marcus-like",
        perspective: "inside",
        description: "Assigns sharply lower probability than median at nearly every node, grounded in deep skepticism that LLMs are on any path toward AGI or systems capable of existential-scale harm.",
        probabilities: {
          "t-ai-inc": 0.1,
          "t-d-no-inc": 0.03,
          "t-single": 0.12,
          "t-d-multi": 0.06,
          "t-has-rep": 0.08,
          "t-d-no-rep": 0.09,
          "t-expects": 0.07,
          "t-d-expects": 0.22,
          "t-d-no-expects": 0.07
        },
        ranges: {
          "t-ai-inc":       { lo: 0.03, hi: 0.22 },
          "t-d-no-inc":       { lo: 0.01, hi: 0.08 },
          "t-single":       { lo: 0.03, hi: 0.28 },
          "t-d-multi":       { lo: 0.02, hi: 0.15 },
          "t-has-rep":       { lo: 0.02, hi: 0.2 },
          "t-d-no-rep":       { lo: 0.02, hi: 0.22 },
          "t-expects":       { lo: 0.01, hi: 0.18 },
          "t-d-expects":       { lo: 0.08, hi: 0.45 },
          "t-d-no-expects":       { lo: 0.02, hi: 0.18 }
        },
        reasoning: {
          "t-ai-inc": "Marcus repeatedly argues that current LLMs are 'high-tech autocomplete' lacking genuine reasoning, and in 'Rebooting AI' and his Substack he frames x-risk concerns as 'premature at best' given the brittleness of current systems; he does allow a small tail for misuse-driven catastrophe (e.g. bioweapons via AI-assisted design) even if he dismisses AGI-style risk.",
          "t-d-no-inc": "Marcus has no particular skepticism about non-AI existential risks (nuclear, pandemic, climate); this is a standard background-rate estimate he would likely defer to mainstream risk analysts on, placing it in the low single digits.",
          "t-single": "Marcus consistently argues that AI development is fragile and contested across many actors; the scenario of one dominant AI system powerful enough to unilaterally determine existential outcomes he treats as science-fiction, inconsistent with how he sees the competitive, multi-lab, multi-national landscape.",
          "t-d-multi": "Even in a multipolar AI landscape, Marcus's capability skepticism leads him to doubt any collection of current-paradigm systems could coordinate or stumble into existential catastrophe; he'd see competitive dynamics as more likely to produce messy harms than existential ones.",
          "t-has-rep": "Marcus is one of the most vocal critics of the idea that LLMs have genuine conceptual representations — he argues they manipulate tokens without grounding them in meaning and that 'understanding' is absent; he would strongly doubt any near-future system develops a stable internal model of an abstract concept like existential catastrophe.",
          "t-d-no-rep": "Marcus acknowledges reward-hacking and side-effects as real near-term problems but treats existential-scale unintended harm as implausible given the limitations of current architectures; he views the systems as too brittle and narrow to cause civilisation-level damage even accidentally.",
          "t-expects": "Marcus has argued that LLMs lack genuine goals or intentions — he distinguishes 'stochastic pattern completion' from purposive agency; he would assign very low probability that any near-future system could form something deserving the label 'expectation' or 'intent' regarding a global catastrophe.",
          "t-d-expects": "This node is already deep in a conditional Marcus thinks is extremely unlikely; conditional on all those improbable steps occurring, he would grant a non-trivial probability of catastrophe since a system that genuinely intends and pursues D is by stipulation highly capable, but his residual capability skepticism still tempers it below 0.5.",
          "t-d-no-expects": "The miscalculation scenario requires a system that has sophisticated beliefs but systematically wrong ones about catastrophic outcomes — Marcus's view that current systems lack robust world-models makes this doubly unlikely; he'd see this path as requiring a level of structured agency he doesn't think scaling produces."
        }
      },
      ng: {
        name: "Ng-like",
        author: "Ng-like",
        perspective: "inside",
        description: "Practitioner AI optimist who treats near-term existential AI risk as near-zero, viewing it as a speculative distraction from real near-term AI benefits and harms.",
        probabilities: {
          "t-ai-inc": 0.05,
          "t-d-no-inc": 0.02,
          "t-single": 0.15,
          "t-d-multi": 0.04,
          "t-has-rep": 0.15,
          "t-d-no-rep": 0.07,
          "t-expects": 0.08,
          "t-d-expects": 0.25,
          "t-d-no-expects": 0.08
        },
        ranges: {
          "t-ai-inc":       { lo: 0.01, hi: 0.12 },
          "t-d-no-inc":       { lo: 0.005, hi: 0.05 },
          "t-single":       { lo: 0.05, hi: 0.3 },
          "t-d-multi":       { lo: 0.01, hi: 0.1 },
          "t-has-rep":       { lo: 0.05, hi: 0.3 },
          "t-d-no-rep":       { lo: 0.02, hi: 0.15 },
          "t-expects":       { lo: 0.02, hi: 0.18 },
          "t-d-expects":       { lo: 0.08, hi: 0.45 },
          "t-d-no-expects":       { lo: 0.02, hi: 0.18 }
        },
        reasoning: {
          "t-ai-inc": "Ng has repeatedly argued AI is a net positive — 'AI is the new electricity' — and explicitly compared existential AI worries to 'worrying about overpopulation on Mars.' He would expect AI to lower x-risk (via medical advances, etc.) at least as much as it raises it, placing a small but nonzero probability on net-negative.",
          "t-d-no-inc": "Ng rarely discusses non-AI existential risks directly, but as a scientist he would not dismiss all existential hazards (pandemics, nuclear war). He would accept a small background rate broadly consistent with mainstream scientific estimates (~1–5% per century scaled to 30 years).",
          "t-single": "Ng has advocated strongly for AI democratization and open-source access (e.g., his criticism of export controls and AI consolidation), implying he thinks concentration of AI power is unlikely; conditional on AI raising x-risk, he would still lean toward multipolar over single-actor scenarios.",
          "t-d-multi": "Ng would view a distributed AI landscape as protective: no single actor has decisive leverage, market incentives self-correct, and governments can intervene. Multipolar catastrophe requires a coordination failure across many systems simultaneously, which he would treat as very unlikely.",
          "t-has-rep": "Ng has publicly stated that current LLMs and near-future AI systems do not have persistent goals or world-models in the sense implied by classical AGI alignment concerns; he would expect most plausible harm worlds to be indirect side-effects rather than a system that internally models catastrophe as a concept.",
          "t-d-no-rep": "Even conceding unaware reward-hacking harm, Ng would argue that engineering feedback loops, regulatory oversight, and the incremental nature of AI deployment make civilisation-ending scale implausible; serious harm is possible but extinction from purely side-effect-driven AI is not.",
          "t-expects": "Ng is deeply skeptical of goal-directed superintelligence emerging on any near-term timescale, calling such scenarios 'science fiction' in multiple interviews. Conditional on a single dominant AI already raising x-risk, he would still put low probability on that system possessing something like deliberate intent toward catastrophe.",
          "t-d-expects": "This is the scenario closest to the classic 'misaligned superintelligence' framing that Ng dismisses — but if forced to condition on a genuinely intent-bearing dominant AI, he would acknowledge that deliberate pursuit of catastrophic outcomes is the most dangerous sub-case and assign a meaningfully higher conditional probability than the other branches.",
          "t-d-no-expects": "Miscalculation by an AI that models danger but holds wrong beliefs is more correctable than deliberate action: humans can observe divergences from intended behavior and retrain or shut down systems. Ng's iterative-deployment worldview suggests this failure mode is detectable and recoverable, warranting a lower conditional than the deliberate case."
        }
      },
      lecun: {
        name: "LeCun-like",
        author: "LeCun-like",
        perspective: "inside",
        description: "Views AI x-risk as essentially zero by rejecting AI agency, instrumental convergence, and the singleton scenario, placing him far below the median on every AI-specific leaf while also being broadly optimistic about non-AI risks.",
        probabilities: {
          "t-ai-inc": 0.005,
          "t-d-no-inc": 0.01,
          "t-single": 0.05,
          "t-d-multi": 0.02,
          "t-has-rep": 0.1,
          "t-d-no-rep": 0.05,
          "t-expects": 0.07,
          "t-d-expects": 0.02,
          "t-d-no-expects": 0.01
        },
        ranges: {
          "t-ai-inc":       { lo: 0.0005, hi: 0.02 },
          "t-d-no-inc":       { lo: 0.003, hi: 0.03 },
          "t-single":       { lo: 0.02, hi: 0.15 },
          "t-d-multi":       { lo: 0.005, hi: 0.08 },
          "t-has-rep":       { lo: 0.03, hi: 0.25 },
          "t-d-no-rep":       { lo: 0.01, hi: 0.15 },
          "t-expects":       { lo: 0.02, hi: 0.2 },
          "t-d-expects":       { lo: 0.005, hi: 0.1 },
          "t-d-no-expects":       { lo: 0.002, hi: 0.05 }
        },
        reasoning: {
          "t-ai-inc": "LeCun has repeatedly called AI doom fears 'preposterously ridiculous' and publicly states P(doom) below 0.01%. He argues AI is overwhelmingly net-beneficial — curing diseases, expanding scientific capability, solving climate problems — and that even deliberate misuse, while non-zero, falls far short of existential scale; the residual here reflects that floor rather than any concession to autonomous-AI risk.",
          "t-d-no-inc": "LeCun is a broad technological optimist who believes human ingenuity and institutional resilience will prevent existential-scale catastrophes irrespective of AI; his public framing consistently implies that non-AI existential risks (nuclear, pandemic, asteroid) are already very low over any 30-year window, even if he rarely foregrounds them explicitly.",
          "t-single": "LeCun explicitly and repeatedly argues that AI development is and will remain competitive and multipolar — involving the US, China, Europe, open-source communities, and countless other actors — making a single dominant AI essentially impossible. He frequently cites open-source AI development as a structural safeguard against monopoly, and has called the 'superintelligent singleton' scenario science fiction.",
          "t-d-multi": "In a competitive multipolar AI landscape, LeCun believes market forces, geopolitical competition, and distributed oversight naturally constrain runaway harm; catastrophic coordination would require a monopoly he views as structurally prevented, so even if AI systems proliferate dangerously, catastrophe-level outcomes remain very unlikely.",
          "t-has-rep": "LeCun consistently argues that current AI — including near-future successors to LLMs — are 'autocomplete' systems lacking genuine understanding, intentionality, or goals. His entire JEPA / objective-driven AI research agenda is premised on building AI that pursues human-specified objectives without developing autonomous representations of dangerous ends; he sees no mechanism by which this would arise spontaneously.",
          "t-d-no-rep": "Even if AI systems produced harm through reward hacking or side-effects without 'understanding' it, LeCun argues that engineering discipline, objective-driven design, and iterative deployment with human oversight would catch and correct such failures well before they reach existential scale; he views this as a solved engineering problem given proper design choices.",
          "t-expects": "LeCun is the most prominent public critic of 'instrumental convergence' arguments — the claim that capable AI would spontaneously develop self-preservation or world-domination goals. He has said this reasoning incorrectly anthropomorphizes AI, that there is no mechanism by which an AI would 'want' to dominate, and has compared such arguments to unfounded science-fiction extrapolation.",
          "t-d-expects": "Even in the extreme and (to LeCun) implausible scenario where an AI foresaw and intended catastrophic outcomes, he would argue that distributed human oversight, redundant control systems, and the competitive multipolar AI landscape would prevent any single system from executing a civilization-ending plan; he sees societal and institutional responses as decisive.",
          "t-d-no-expects": "A miscalculating AI causing existential harm requires autonomous agency at massive scale plus an absence of engineering safeguards — both of which his objective-driven AI agenda is explicitly designed to prevent. He views safety as a tractable engineering problem with known solution paths, not an open philosophical puzzle, and therefore assigns low probability even to this accidental world."
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
