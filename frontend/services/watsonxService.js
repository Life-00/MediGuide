// MediGuide 백엔드 연동 서비스 (RAG 브랜치 대응)

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
const CHAT_ENDPOINT = `${BACKEND_URL}/chat`;
const SUGGESTIONS_ENDPOINT = `${BACKEND_URL}/suggested_questions`;

export class WatsonxService {
  constructor() {
    this.activeSessionId = null;
  }

  // 새로운 채팅 세션 생성
  createNewChat() {
    this.activeSessionId = `session_${Date.now()}`;
    return this.activeSessionId;
  }

  // 현재 세션 ID 반환
  getSessionId() {
    if (!this.activeSessionId) {
      this.createNewChat();
    }
    return this.activeSessionId;
  }

  // 메시지 전송 (통합 엔드포인트)
  async sendMessage(message) {
    if (!this.activeSessionId) {
      this.createNewChat();
    }

    console.log('🚀 [MediGuide] 요청:', CHAT_ENDPOINT);
    console.log('📤 [MediGuide] 데이터:', { query: message, session_id: this.activeSessionId });

    try {
      const response = await fetch(CHAT_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: message,
          session_id: this.activeSessionId,
        }),
      });

      console.log('📊 [MediGuide] 응답 상태:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ [MediGuide] 에러:', errorText);
        throw new Error(`백엔드 연결 실패! 상태 코드: ${response.status}`);
      }

      const data = await response.json();
      console.log('✅ [MediGuide] 응답 타입:', data.type);

      return data;
      // 응답 형식:
      // {
      //   answer: string,
      //   type: "chat" | "document" | "error",
      //   document_content: string | null,
      //   sources: Array
      // }

    } catch (error) {
      console.error('❌ [MediGuide] 에러:', error);
      throw error;
    }
  }

  // 추천 질문 가져오기
  async getSuggestedQuestions() {
    try {
      const response = await fetch(SUGGESTIONS_ENDPOINT);
      if (!response.ok) {
        throw new Error('추천 질문을 불러올 수 없습니다.');
      }
      const data = await response.json();
      return data.questions || [];
    } catch (error) {
      console.error('❌ 추천 질문 로딩 실패:', error);
      return [];
    }
  }
}

export const watsonxService = new WatsonxService();