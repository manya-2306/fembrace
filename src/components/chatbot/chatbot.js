import React, { useState, useRef, useEffect } from "react";
import "./chatbot.css";

const Chatbot = ({ result }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const chatEndRef = useRef(null);

  // Auto-scroll to latest message
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, loading]);

  const handleSend = async () => {
    if (!message.trim()) return;

    const userMessage = message;

    // Add user message to chat history
    setMessages((prev) => [
      ...prev,
      {
        sender: "user",
        text: userMessage,
      },
    ]);

    // Clear input immediately
    setMessage("");
    setLoading(true);

    try {
      const response = await fetch(
        "http://127.0.0.1:5002/chatbot",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: userMessage,
            prediction: result?.prediction || "Unknown",
            probability: result?.probability || 0,
            top_features: result?.top_features || [],
            form_data: result?.formData || {},
            explanation: result?.explanation || {},
            }),
        }
      );

      const data = await response.json();

      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: (data.reply || "No response received.")
        .replace(/\*\*/g, "")
        .replace(/\*/g, "")
        .replace(/#+/g, "")
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text:
            "Something went wrong while contacting FemBrace AI.",
        },
      ]);
    }

    setLoading(false);
  };

  return (
    <>
      {/* Floating Button */}
      <button
        className="chatbot-floating-btn"
        onClick={() => setIsOpen(!isOpen)}
      >
        💬 Ask FemBrace AI
      </button>

      {/* Chat Popup */}
      {isOpen && (
        <div className="chatbot-popup">
          <div className="chatbot-header">
            <h3>FemBrace AI Assistant</h3>

            <button
              className="close-btn"
              onClick={() =>
                setIsOpen(false)
              }
            >
              ✕
            </button>
          </div>

          <div className="chatbot-body">
            {/* Chat History */}
            <div className="chat-history">
              {messages.length === 0 && (
                <div className="chatbot-welcome">
                  Ask about PCOS, symptoms,
                  prediction, diet, or lifestyle
                  improvements.
                </div>
              )}

              {messages.map((msg, index) => (
                <div
                  key={index}
                  className={`chat-message ${
                    msg.sender === "user"
                      ? "user-message"
                      : "bot-message"
                  }`}
                >
                  <p>{msg.text}</p>
                </div>
              ))}

              {loading && (
                <div className="chat-message bot-message">
                  <p>Thinking...</p>
                </div>
              )}

              <div ref={chatEndRef} />
            </div>

            {/* Input */}
            <textarea
              placeholder="Ask FemBrace AI..."
              value={message}
              onChange={(e) =>
                setMessage(e.target.value)
              }
              rows="3"
            />

            <button
              className="send-btn"
              onClick={handleSend}
              disabled={loading}
            >
              Send
            </button>
          </div>
        </div>
      )}
    </>
  );
};

export default Chatbot;