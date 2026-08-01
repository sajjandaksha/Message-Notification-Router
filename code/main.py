import argparse
import pandas as pd
from pathlib import Path
from router import MessageRouter


def main():
    parser = argparse.ArgumentParser(
        description="WhatsApp Message Notification Router"
    )

    parser.add_argument(
        "--dataset",
        default="dataset",
        help="Dataset folder"
    )

    parser.add_argument(
        "--output",
        default="dataset/output.csv",
        help="Output CSV path"
    )

    args = parser.parse_args()

    dataset = Path(args.dataset)

    messages = pd.read_csv(dataset / "messages.csv")

    users = pd.read_csv(dataset / "users.csv")

    groups = pd.read_csv(dataset / "groups.csv")

    group_members = pd.read_csv(dataset / "group_members.csv")

    businesses = pd.read_csv(dataset / "business_accounts.csv")

    business_history = pd.read_csv(dataset / "user_business_history.csv")

    history = pd.read_csv(dataset / "message_history.csv")

    events = pd.read_csv(dataset / "message_events.csv")

    images = pd.read_csv(dataset / "images.csv")

    voice = pd.read_csv(dataset / "voice_notes.csv")

    summary = pd.read_csv(dataset / "daily_notification_summary.csv")

    router = MessageRouter(
        users,
        groups,
        group_members,
        businesses,
        business_history,
        history,
        events,
        images,
        voice,
        summary,
        dataset
    )

    predictions = []

    for _, row in messages.iterrows():
        result = router.process_message(row)
        predictions.append(result)

    output = pd.DataFrame(predictions)

    output.to_csv(args.output, index=False)

    print(f"Finished! Saved to {args.output}")


if __name__ == "__main__":
    main()
