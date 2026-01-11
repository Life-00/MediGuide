// Watsonx 백엔드 연동 서비스

// 백엔드 서버 URL 설정 (Vite 환경변수 사용)
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

// TODO: 백엔드 개발 완료 후 정확한 경로로 수정 필요!
// 임시로 /api/chat 사용 중
const CHAT_ENDPOINT = `${BACKEND_URL}/api/chat`;

export class WatsonxService {
  constructor() {
    this.activeSessionId = null;
  }

  // 새로운 채팅 세션 생성
  createNewChat() {
    this.activeSessionId = Date.now().toString();
    return this.activeSessionId;
  }

  // 스트리밍 방식으로 메시지 전송
  async *sendMessageStream(message) {
    if (!this.activeSessionId) {
      this.createNewChat();
    }

    console.log('🚀 [DEBUG] 요청 시작:', CHAT_ENDPOINT);
    console.log('📤 [DEBUG] 보내는 데이터:', { message, sessionId: this.activeSessionId });

    try {
      // 백엔드 서버에 POST 요청
      const response = await fetch(CHAT_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // TODO: 백엔드 개발 완료 후 API 키 처리 방식 확인 필요
          // 백엔드에서 자체 처리 예정
        },
        body: JSON.stringify({
          // TODO: 백엔드 개발 완료 후 정확한 필드명으로 수정 필요
          // 임시로 message, sessionId 사용 중
          message: message,
          sessionId: this.activeSessionId,
        }),
      });

      console.log('📊 [DEBUG] 응답 상태:', response.status, response.statusText);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ [DEBUG] 에러 응답:', errorText);
        throw new Error(`백엔드 연결 실패! 상태 코드: ${response.status}`);
      }

      // === 스트리밍 응답 처리 ===
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      console.log('📥 [DEBUG] 스트리밍 시작...');
      let chunkCount = 0;

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          console.log('✅ [DEBUG] 스트리밍 완료! 총 청크:', chunkCount);
          break;
        }
        
        chunkCount++;
        
        // 받은 데이터를 텍스트로 변환
        const chunk = decoder.decode(value, { stream: true });
        console.log(`📦 [DEBUG] 청크 #${chunkCount}:`, chunk.substring(0, 100));
        
        // 백엔드 응답 형식에 따라 아래 코드 수정 필요!
        // 
        // 형식 1: Server-Sent Events (SSE) - "data: {...}\n" 형식
        const lines = chunk.split('\n').filter(line => line.trim());
        
        for (const line of lines) {
          // "data: " 접두사 제거
          let data = line.replace(/^data:\s*/, '');
          
          // [DONE] 시그널 체크
          if (data === '[DONE]') {
            console.log('🏁 [DEBUG] [DONE] 시그널 받음');
            return;
          }
          
          try {
            // JSON 파싱 시도
            const parsed = JSON.parse(data);
            console.log('🔍 [DEBUG] 파싱된 데이터:', parsed);
            
            // TODO: 백엔드 개발 완료 후 정확한 응답 형식으로 수정 필요!
            // 현재는 여러 형식을 모두 시도하도록 설정
            
            let yielded = false;
            
            // 예시 1: { text: "응답내용" }
            if (parsed.text) {
              console.log('✨ [DEBUG] text 필드 발견:', parsed.text.substring(0, 50));
              yield parsed.text;
              yielded = true;
            }
            // 예시 2: { delta: "응답내용" }
            else if (parsed.delta) {
              console.log('✨ [DEBUG] delta 필드 발견:', parsed.delta.substring(0, 50));
              yield parsed.delta;
              yielded = true;
            }
            // 예시 3: { content: "응답내용" }
            else if (parsed.content) {
              console.log('✨ [DEBUG] content 필드 발견:', parsed.content.substring(0, 50));
              yield parsed.content;
              yielded = true;
            }
            // 예시 4: { response: "응답내용" }
            else if (parsed.response) {
              console.log('✨ [DEBUG] response 필드 발견:', parsed.response.substring(0, 50));
              yield parsed.response;
              yielded = true;
            }
            // 예시 5: Watsonx 특정 형식
            else if (parsed.results && parsed.results[0]?.generated_text) {
              console.log('✨ [DEBUG] results 필드 발견:', parsed.results[0].generated_text.substring(0, 50));
              yield parsed.results[0].generated_text;
              yielded = true;
            }
            
            if (!yielded) {
              console.warn('⚠️ [DEBUG] 알 수 없는 응답 형식! parsed:', parsed);
            }
          } catch (e) {
            // JSON이 아닌 경우 그냥 텍스트로 처리
            if (data.trim()) {
              console.log('📝 [DEBUG] 일반 텍스트로 처리:', data.substring(0, 100));
              yield data;
            }
          }
        }
      }
    } catch (error) {
      console.error('❌ 스트리밍 에러:', error);
      throw error;
    }
  }

  // 스트리밍을 지원하지 않는 경우 (일반 요청/응답)
  async sendMessage(message) {
    if (!this.activeSessionId) {
      this.createNewChat();
    }

    try {
      const response = await fetch(CHAT_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          sessionId: this.activeSessionId,
        }),
      });

      if (!response.ok) {
        throw new Error(`백엔드 연결 실패! 상태 코드: ${response.status}`);
      }

      const data = await response.json();
      
      // 백엔드 응답에서 텍스트 추출 (형식에 맞게 수정)
      return data.response || data.text || data.content || '';
    } catch (error) {
      console.error('❌ 메시지 전송 에러:', error);
      throw error;
    }
  }
}

export const watsonxService = new WatsonxService();