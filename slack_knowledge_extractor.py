import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# --- Configuration ---
load_dotenv()
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TARGET_CHANNEL_NAME = (
    "mehmet-test-ai"  # <--- CHANGE THIS to the exact name of your channel
)
OUTPUT_CSV_FILE = "slack_knowledge_base.csv"
REQUEST_TIMEOUT = 20  # seconds for fetching web pages
MAX_CONTENT_LENGTH = 8000  # Max characters of website content to send to OpenAI
OPENAI_MODEL = "gpt-4o-mini"  # Or "gpt-4" or "gpt-4-turbo" etc.
MESSAGES_LIMIT_PER_PAGE = (
    200  # Max allowed by Slack API is 1000, but smaller can be safer for rate limits
)
RATE_LIMIT_DELAY = 1.2  # Seconds to wait between Slack API calls (adjust based on tier)

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Initialize Clients ---
if not SLACK_BOT_TOKEN:
    raise ValueError("SLACK_BOT_TOKEN not found in environment variables.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

slack_client = WebClient(token=SLACK_BOT_TOKEN)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
user_cache: Dict[str, str] = {}  # Cache for user ID -> user name mapping

# --- Helper Functions ---


def find_channel_id(channel_name: str) -> Optional[str]:
    """Finds the Slack channel ID given its name."""
    logging.info(f"Attempting to find channel ID for '{channel_name}'...")
    try:
        # Try public channels first
        for result in slack_client.conversations_list(limit=1000):
            for channel in result["channels"]:
                if channel["name"] == channel_name:
                    logging.info(f"Found public channel ID: {channel['id']}")
                    return channel["id"]
            time.sleep(RATE_LIMIT_DELAY)  # Respect rate limits

        # If not found, try private channels (requires groups:read scope)
        try:
            for result in slack_client.conversations_list(
                types="private_channel", limit=1000
            ):
                for channel in result["channels"]:
                    if channel["name"] == channel_name:
                        logging.info(f"Found private channel ID: {channel['id']}")
                        return channel["id"]
                time.sleep(RATE_LIMIT_DELAY)
        except SlackApiError as e:
            if e.response["error"] == "missing_scope":
                logging.warning(
                    "Missing 'groups:read' scope to search private channels."
                )
            else:
                raise e  # Re-raise other Slack API errors

    except SlackApiError as e:
        logging.error(f"Error fetching conversations: {e.response['error']}")
    return None


def get_user_name(user_id: str) -> str:
    """Fetches user name from Slack API, using a cache."""
    if user_id in user_cache:
        return user_cache[user_id]
    try:
        result = slack_client.users_info(user=user_id)
        user_name = (
            result["user"]["real_name"] or result["user"]["name"]
        )  # Prefer real name
        user_cache[user_id] = user_name
        logging.debug(f"Fetched user info for {user_id}: {user_name}")
        time.sleep(RATE_LIMIT_DELAY)  # Be nice to the User API rate limits too
        return user_name
    except SlackApiError as e:
        logging.error(f"Error fetching user info for {user_id}: {e.response['error']}")
        user_cache[user_id] = f"Unknown User ({user_id})"  # Cache the error state
        return user_cache[user_id]
    except Exception as e:
        logging.error(f"Unexpected error fetching user info for {user_id}: {e}")
        user_cache[user_id] = f"Unknown User ({user_id})"
        return user_cache[user_id]


def get_all_messages(channel_id: str) -> List[Dict[str, Any]]:
    """Fetches all messages and thread replies from a channel."""
    all_messages: List[Dict[str, Any]] = []
    processed_thread_ts: set[str] = set()
    next_cursor = None
    message_count = 0

    logging.info(f"Fetching messages from channel {channel_id}...")

    try:
        while True:
            logging.info(f"Fetching history page (cursor: {next_cursor})...")
            response = slack_client.conversations_history(
                channel=channel_id, limit=MESSAGES_LIMIT_PER_PAGE, cursor=next_cursor
            )
            messages = response.get("messages", [])
            all_messages.extend(messages)
            message_count += len(messages)
            logging.info(
                f"Fetched {len(messages)} messages. Total so far: {message_count}"
            )

            # Process threads originating from this batch of messages
            for message in messages:
                thread_ts = message.get("thread_ts")
                # Ensure it's the *start* of a thread and hasn't been processed
                if (
                    thread_ts == message.get("ts")
                    and message.get("reply_count", 0) > 0
                    and thread_ts not in processed_thread_ts
                ):
                    logging.info(f"Fetching replies for thread {thread_ts}...")
                    thread_replies = get_thread_replies(channel_id, thread_ts)
                    all_messages.extend(thread_replies)  # Add replies to the main list
                    processed_thread_ts.add(thread_ts)  # Mark thread as processed
                    message_count += len(thread_replies)
                    logging.info(
                        f"Fetched {len(thread_replies)} replies for thread {thread_ts}. Total messages: {message_count}"
                    )

            if not response.get("has_more"):
                break
            next_cursor = response.get("response_metadata", {}).get("next_cursor")
            time.sleep(RATE_LIMIT_DELAY)  # Wait before fetching next page

    except SlackApiError as e:
        logging.error(f"Error fetching channel history: {e.response['error']}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during message fetching: {e}")

    logging.info(
        f"Finished fetching. Total messages including replies: {len(all_messages)}"
    )
    # Sort messages by timestamp (optional, but often useful)
    all_messages.sort(key=lambda m: float(m.get("ts", 0)))
    return all_messages


def get_thread_replies(channel_id: str, thread_ts: str) -> List[Dict[str, Any]]:
    """Fetches all replies for a specific thread."""
    thread_messages: List[Dict[str, Any]] = []
    next_cursor = None
    try:
        while True:
            logging.debug(
                f"Fetching replies page for {thread_ts} (cursor: {next_cursor})..."
            )
            response = slack_client.conversations_replies(
                channel=channel_id,
                ts=thread_ts,
                limit=MESSAGES_LIMIT_PER_PAGE,
                cursor=next_cursor,
            )
            messages = response.get("messages", [])
            # The first message is the parent, skip it if we already have it
            thread_messages.extend(messages[1:] if not next_cursor else messages)

            if not response.get("has_more"):
                break
            next_cursor = response.get("response_metadata", {}).get("next_cursor")
            time.sleep(RATE_LIMIT_DELAY)

    except SlackApiError as e:
        logging.error(
            f"Error fetching thread replies for {thread_ts}: {e.response['error']}"
        )
    except Exception as e:
        logging.error(f"An unexpected error occurred during thread reply fetching: {e}")

    return thread_messages


def extract_links(text: str) -> List[str]:
    """Extracts URLs from text."""
    if not text:
        return []
    # Slack often wraps links in < >
    slack_link_pattern = r"<(https?://[^>|]+)(?:\|[^>]+)?>"
    # General URL pattern (might catch some false positives)
    general_link_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'

    slack_links = [match.group(1) for match in re.finditer(slack_link_pattern, text)]
    # Find general links that weren't already found as Slack links
    general_links_found = re.findall(general_link_pattern, text)
    unique_general_links = [
        link
        for link in general_links_found
        if link not in slack_links and not any(s_link in link for s_link in slack_links)
    ]

    return slack_links + unique_general_links


def get_webpage_content(url: str) -> Optional[str]:
    """Fetches and extracts text content from a webpage."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(
            url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True
        )
        response.raise_for_status()  # Raise an exception for bad status codes

        # Check content type - only parse HTML
        content_type = response.headers.get("content-type", "").lower()
        if "html" not in content_type:
            logging.warning(
                f"Skipping non-HTML content at {url} (type: {content_type})"
            )
            return None

        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Get text, trying common main content tags first
        main_content = soup.find("main") or soup.find("article") or soup.find("body")
        text = ""
        if main_content:
            text = main_content.get_text(separator=" ", strip=True)
        else:
            text = soup.get_text(separator=" ", strip=True)  # Fallback to whole body

        # Limit length to avoid huge OpenAI prompts
        return text[:MAX_CONTENT_LENGTH]

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching URL {url}: {e}")
    except Exception as e:
        logging.error(f"Error parsing URL {url}: {e}")
    return None


def analyze_content_with_ai(
    content: str, source_type: str = "Web Page Content"
) -> Dict[str, str]:
    """Uses OpenAI to analyze text for purpose, pricing, and comments."""
    prompt = f"""
    Analyze the following {source_type}:
    --- START CONTENT ---
    {content}
    --- END CONTENT ---

    Based *only* on the text provided above, extract the following information about the tool, product, or concept described:
    1. Purpose: A brief description of what it is or does.
    2. Pricing: Mention any specific pricing tiers, free plans, or costs mentioned. If no pricing is explicitly mentioned, state "Not specified".
    3. Comment: Any other interesting features, target audience, or key takeaways. If nothing else relevant, state "N/A".

    Provide the output as a JSON object with the keys "Purpose", "Pricing", and "Comment".
    Example: {{ "Purpose": "...", "Pricing": "...", "Comment": "..." }}
    If the content is irrelevant or doesn't describe a tool/product/concept, return:
    {{ "Purpose": "N/A", "Pricing": "N/A", "Comment": "Content does not describe a specific tool or concept." }}
    """

    logging.info(
        f"Sending content (approx {len(content)} chars) to OpenAI for analysis..."
    )
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant tasked with summarizing information about tools and concepts from provided text.",
                },
                {"role": "user", "content": prompt},
            ],
            # response_format={"type": "json_object"}, # Uncomment if using newer models supporting JSON mode
            temperature=0.2,  # Lower temperature for more factual extraction
            max_tokens=300,  # Adjust as needed
        )
        ai_response_text = response.choices[0].message.content
        logging.info("Received analysis from OpenAI.")
        logging.debug(f"OpenAI raw response: {ai_response_text}")

        # Attempt to parse the JSON (handle potential non-JSON responses)
        import json

        try:
            # Find the JSON part in case the model adds extra text
            json_match = re.search(r"\{.*\}", ai_response_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group(0))
                # Validate expected keys
                if all(k in analysis for k in ["Purpose", "Pricing", "Comment"]):
                    return {
                        "Purpose": analysis.get("Purpose", "Error parsing response"),
                        "Pricing": analysis.get("Pricing", "Error parsing response"),
                        "Comment": analysis.get("Comment", "Error parsing response"),
                    }
                else:
                    logging.warning("OpenAI response JSON missing expected keys.")
                    return {
                        "Purpose": "Error: Invalid JSON structure",
                        "Pricing": "N/A",
                        "Comment": ai_response_text or "Empty response",
                    }
            else:
                logging.warning("Could not find JSON object in OpenAI response.")
                # Fallback: Try to extract info manually (less reliable)
                purpose = re.search(r"Purpose:\s*(.*)", ai_response_text, re.IGNORECASE)
                pricing = re.search(r"Pricing:\s*(.*)", ai_response_text, re.IGNORECASE)
                comment = re.search(r"Comment:\s*(.*)", ai_response_text, re.IGNORECASE)
                return {
                    "Purpose": (
                        purpose.group(1).strip()
                        if purpose
                        else "Parsing fallback failed"
                    ),
                    "Pricing": (
                        pricing.group(1).strip()
                        if pricing
                        else "Parsing fallback failed"
                    ),
                    "Comment": (
                        comment.group(1).strip()
                        if comment
                        else "Parsing fallback failed"
                    ),
                }

        except json.JSONDecodeError:
            logging.error(
                f"Failed to decode JSON from OpenAI response: {ai_response_text}"
            )
            return {
                "Purpose": "Error: Invalid JSON response",
                "Pricing": "N/A",
                "Comment": ai_response_text or "Empty response",
            }

    except RateLimitError:
        logging.error(
            "OpenAI Rate Limit Exceeded. Please wait and try again, or check your plan."
        )
        return {
            "Purpose": "Error: OpenAI Rate Limit",
            "Pricing": "N/A",
            "Comment": "N/A",
        }
    except (APIError, APITimeoutError) as e:
        logging.error(f"OpenAI API Error: {e}")
        return {
            "Purpose": f"Error: OpenAI API Error ({type(e).__name__})",
            "Pricing": "N/A",
            "Comment": "N/A",
        }
    except Exception as e:
        logging.error(f"Unexpected error during OpenAI call: {e}")
        return {
            "Purpose": f"Error: Unexpected ({type(e).__name__})",
            "Pricing": "N/A",
            "Comment": "N/A",
        }


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Starting Slack Knowledge Extractor...")

    channel_id = find_channel_id(TARGET_CHANNEL_NAME)
    if not channel_id:
        logging.error(
            f"Channel '{TARGET_CHANNEL_NAME}' not found or bot lacks permissions. Exiting."
        )
        exit()

    all_messages = get_all_messages(channel_id)
    if not all_messages:
        logging.warning("No messages found in the channel.")
        exit()

    knowledge_base = []
    processed_items_count = 0

    for message in all_messages:
        # Skip messages without text or from bots (optional)
        msg_text = message.get("text")
        if not msg_text or message.get("subtype") == "bot_message":
            continue

        user_id = message.get("user")
        shared_by = get_user_name(user_id) if user_id else "Unknown User"
        timestamp = datetime.fromtimestamp(float(message.get("ts", 0))).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        links = extract_links(msg_text)

        analysis_results = []  # Store results for this message

        if links:
            logging.info(
                f"Found {len(links)} link(s) in message from {shared_by} at {timestamp}: {links}"
            )
            for link in links:
                # Basic check to avoid analyzing common/uninteresting links (customize as needed)
                if any(
                    domain in link
                    for domain in ["slack.com", "google.com/url?", "yourcompany.com"]
                ):
                    logging.info(f"Skipping common link: {link}")
                    continue

                logging.info(f"Processing link: {link}")
                content = get_webpage_content(link)
                if content:
                    analysis = analyze_content_with_ai(
                        content, source_type="Web Page Content"
                    )
                    analysis_results.append(
                        {
                            "Link": link,
                            "Source Text": None,  # Mark that it came from a link
                            **analysis,  # Unpack Purpose, Pricing, Comment
                            "Shared By": shared_by,
                            "Timestamp": timestamp,
                        }
                    )
                    processed_items_count += 1
                    time.sleep(
                        1
                    )  # Small delay before next OpenAI call if processing multiple links
                else:
                    logging.warning(f"Could not retrieve content for link: {link}")
                    analysis_results.append(
                        {
                            "Link": link,
                            "Source Text": None,
                            "Purpose": "Error: Could not fetch content",
                            "Pricing": "N/A",
                            "Comment": "N/A",
                            "Shared By": shared_by,
                            "Timestamp": timestamp,
                        }
                    )
                    processed_items_count += 1  # Still count it as processed

        # If no *relevant* links were processed, consider the text itself as knowledge
        elif not links:  # Only analyze text if no links were found at all
            # Heuristic: Only analyze messages longer than N chars, or containing keywords? (optional)
            if (
                len(msg_text.strip()) > 50
            ):  # Example: Only analyze reasonably long text snippets
                logging.info(
                    f"No links found. Analyzing message text from {shared_by} at {timestamp} as potential knowledge."
                )
                analysis = analyze_content_with_ai(
                    msg_text, source_type="Slack Message Text"
                )
                # Only add if AI found something potentially useful
                if analysis.get(
                    "Purpose"
                ) != "N/A" and "Content does not describe" not in analysis.get(
                    "Comment", ""
                ):
                    analysis_results.append(
                        {
                            "Link": None,  # Mark that it came from text
                            "Source Text": (
                                msg_text[:200] + "..."
                                if len(msg_text) > 200
                                else msg_text
                            ),  # Store snippet
                            **analysis,
                            "Shared By": shared_by,
                            "Timestamp": timestamp,
                        }
                    )
                    processed_items_count += 1
                else:
                    logging.info(
                        "AI analysis deemed the text not relevant as standalone knowledge."
                    )
            else:
                logging.debug(f"Skipping short message text from {shared_by}")

        # Add all results from this message to the main knowledge base
        knowledge_base.extend(analysis_results)

        # Optional: Add a small delay between processing messages to be safe
        # time.sleep(0.1)

    logging.info(
        f"Finished processing messages. Found {len(knowledge_base)} potential knowledge items from {processed_items_count} analyzed links/texts."
    )

    # --- Save to CSV ---
    if knowledge_base:
        df = pd.DataFrame(knowledge_base)
        # Reorder columns for better readability
        df = df[
            [
                "Timestamp",
                "Shared By",
                "Link",
                "Purpose",
                "Pricing",
                "Comment",
                "Source Text",
            ]
        ]
        try:
            df.to_csv(OUTPUT_CSV_FILE, index=False, encoding="utf-8")
            logging.info(f"Knowledge base successfully saved to {OUTPUT_CSV_FILE}")
        except Exception as e:
            logging.error(f"Error saving knowledge base to CSV: {e}")
    else:
        logging.info("No relevant knowledge items found to save.")

    logging.info("Script finished.")
