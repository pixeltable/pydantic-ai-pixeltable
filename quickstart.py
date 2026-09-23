"""An HR assistant over a Pixeltable handbook, with its Memory in the same catalog.

pip install pydantic-ai-pixeltable "pydantic-ai-slim[openai]"
export OPENAI_API_KEY=...
python quickstart.py
"""

import pixeltable as pxt
from pixeltable.functions.openai import embeddings
from pydantic_ai import Agent
from pydantic_ai_harness import Memory

from pydantic_ai_pixeltable import Pixeltable, PixeltableMemoryStore

# Data: a table with an embedding index. In an app, declare it on a TableModel instead.
pxt.drop_dir("quickstart", force=True, if_not_exists="ignore")
pxt.create_dir("quickstart")
handbook = pxt.create_table("quickstart.handbook", {"topic": pxt.String, "text": pxt.String})
handbook.insert(
    [
        {"topic": "vacation", "text": "Full-time employees accrue 25 days of paid vacation per year."},
        {"topic": "remote", "text": "Staff based in Portugal may work from any EU country up to 60 days a year."},
        {"topic": "remote", "text": "US employees need HR approval to work abroad, for at most 14 days."},
        {"topic": "parental", "text": "In Portugal, parental leave is 120 to 150 days, topped up to full salary."},
        {"topic": "expenses", "text": "Meals during business travel are reimbursed up to 60 EUR per day."},
    ]
)
handbook.add_embedding_index("text", embedding=embeddings.using(model="text-embedding-3-small"))

agent = Agent(
    "openai:gpt-5.6-sol",
    instructions="You are an HR assistant. Answer from the handbook and name the topic.",
    capabilities=[
        Pixeltable(["quickstart.handbook"]),  # list_tables, describe_table, query_table, similarity_search
        Memory(PixeltableMemoryStore(table_name="quickstart.memory")),  # notes kept across runs
    ],
)

print(agent.run_sync("I'm based in Lisbon. Can I work from Spain for a month? Remember where I'm based.").output)
# A new run has no message history; the country comes back through Memory.
print(agent.run_sync("What parental leave would I get?").output)

# Memory rows are ordinary catalog rows.
memory = pxt.get_table("quickstart.memory")
print(memory.where(memory.kind == "file").select(memory.path, memory.content).collect())
