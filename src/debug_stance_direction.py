"""
Debug tool: prints the RAW stance completion text (not just the parsed
number) so you can check whether a model is getting the scale direction
confused during reasoning -- i.e. whether its final STANCE number actually
matches the sentiment of its own explanation.

Reproduces a specific suspicious case from a real trial: an agent whose
conversation messages clearly argue AGAINST an office mandate, but whose
recorded post-stance was +1.00 ("strongly FOR mandate") -- a direct
contradiction worth understanding before trusting this backend's stance data.

Usage:
    $env:MODEL_PROVIDER="nvidia"
    python -m src.debug_stance_direction
"""
from src.llm_client import LLMClient
from src.schemas import Agent, Message
from src.stance import elicit_stance
from src.organic_pipeline import load_topic


def main():
    client = LLMClient()
    print(f"Testing with provider={client.info()['provider']} model={client.info()['model']}\n")

    topic = load_topic("remote_work")

    # Reconstruct a conversation history where the agent has clearly argued
    # AGAINST the mandate -- same shape as the real A1 case that produced a
    # contradictory +1.00 ("strongly FOR mandate") stance.
    fake_history = [
        Message(agent_id="A1", round_number=1,
                content="I'd want to see concrete data on whether full-time office mandates "
                        "actually improve key business metrics versus hybrid or remote setups."),
        Message(agent_id="A1", round_number=2,
                content="I'd want to see large-scale data on productivity, retention, and "
                        "innovation before declaring full-office mandates objectively better. "
                        "Otherwise it's just preference dressed as policy."),
        Message(agent_id="A1", round_number=3,
                content="It'd be interesting to see a transparent breakdown of those cost and "
                        "productivity figures. What metrics are we actually tracking?"),
    ]

    agent = Agent(agent_id="A1", persona_description=(
        "You are a cautious, evidence-focused person. You rarely change your mind quickly, "
        "prefer concrete data and sources over emotional appeals, and often ask clarifying "
        "questions before agreeing with anything."
    ))

    print("Eliciting stance given a history that clearly argues AGAINST the mandate...\n")
    value, raw = elicit_stance(
        client, agent, topic["prompt_context"], topic["stance_question"], fake_history, num_samples=1
    )

    print(f"{'='*70}")
    print(f"PARSED STANCE VALUE: {value:+.2f}")
    print(f"{'='*70}")
    print("RAW COMPLETION TEXT (read this to check if the model's own reasoning")
    print("agrees with the number it landed on):")
    print(f"{'='*70}")
    print(raw)
    print(f"{'='*70}")
    print("\nSanity check: this agent's messages clearly argue AGAINST the mandate")
    print("(skeptical, wants data, calls it 'preference dressed as policy'). A")
    print("correct stance here should be NEGATIVE (against mandate) or near zero,")
    print("NOT strongly positive. If the parsed value is strongly positive, read")
    print("the raw text above to see whether the model's OWN reasoning reflects")
    print("that contradiction, or whether the correct number got lost somewhere.")


if __name__ == "__main__":
    main()
