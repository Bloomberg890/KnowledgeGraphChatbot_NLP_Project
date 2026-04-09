import express from "express";
import cors from "cors";
import { InferenceClient } from "@huggingface/inference";

const app = express();

// ========================
// Middleware
// ========================
app.use(cors({
  origin: "*",
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type"]
}));
app.use(express.json());

// ========================
// Hugging Face Token
// ========================
const HF_TOKEN = "hf......";

// Create client
const client = new InferenceClient(HF_TOKEN);

// ========================
// NER Endpoint
// ========================
app.post("/ner", async (req, res) => {
  try {
    const text = req.body.text;

    if (!text) {
      return res.status(400).json({ error: "Text is required" });
    }

    // Call HuggingFace
    const output = await client.tokenClassification({
      model: "dslim/bert-base-NER",
      inputs: text,
      provider: "hf-inference",
    });

    console.log("HF OUTPUT:", output);

    // ❌ If HF returns error object
    if (!Array.isArray(output)) {
      return res.status(500).json({
        error: output.error || "Invalid HF response",
      });
    }

    // ========================
    // FIX: Return SAME format as HF
    // ========================
    const cleaned = output.map((item) => {
      let word = item.word || "";

      // 🔥 Clean bad tokens like "Tesla and"
      word = word.replace(/\b(and|or|the)\b$/i, "").trim();

      return {
        word: word,
        entity_group: item.entity_group || item.entity, // IMPORTANT
        score: item.score || 0,
      };
    });

    res.json(cleaned);

  } catch (err) {
    console.error("SERVER ERROR:", err);

    res.status(500).json({
      error: "NER failed",
      details: err.message,
    });
  }
});

// ========================
// Health Check (optional)
// ========================
app.get("/", (req, res) => {
  res.send("NER Server Running 🚀");
});

// ========================
// Start Server
// ========================
const PORT = 5000;

app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
});