import { GoogleGenAI } from "@google/genai";

// Vite에서는 환경변수를 import.meta.env로 읽고,
// 노출하려면 반드시 VITE_ 접두사가 필요합니다.
const API_KEY = import.meta.env.VITE_API_KEY || "";

export class GeminiService {
  constructor() {
    this.ai = new GoogleGenAI({ apiKey: API_KEY });
    this.activeChat = null;
  }

  createNewChat() {
    this.activeChat = this.ai.chats.create({
      model: "gemini-3-flash-preview",
      config: {
        systemInstruction:
          "You are a professional, helpful, and concise AI assistant. You provide high-quality, accurate information in a clean format.",
        temperature: 0.7,
        topP: 0.95,
      },
    });

    return this.activeChat;
  }

  async *sendMessageStream(message) {
    if (!this.activeChat) {
      this.createNewChat();
    }

    try {
      const stream = await this.activeChat.sendMessageStream({ message });

      for await (const chunk of stream) {
        // chunk 구조가 라이브러리 버전에 따라 다를 수 있어 안전하게 처리
        yield chunk?.text || "";
      }
    } catch (error) {
      console.error("Error in streaming message:", error);
      throw error;
    }
  }
}

export const geminiService = new GeminiService();
