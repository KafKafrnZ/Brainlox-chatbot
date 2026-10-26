"""Fails if course ranking breaks. No network, no API key."""

from chatbot import rank, reply

CHUNKS = [
    {"text": "Python for beginners covers syntax and small projects", "source": "python"},
    {"text": "Excel dashboards use pivot tables and charts", "source": "excel"},
]


def main() -> None:
    top = rank("how do I learn python syntax", CHUNKS)
    assert top and top[0]["source"] == "python", top
    assert rank("zzzqqq", CHUNKS) == []
    out = reply("python syntax", CHUNKS)
    assert out["mode"] == "retrieve"
    assert "Python" in out["response"]
    print("ok")


if __name__ == "__main__":
    main()
