import React, { useMemo, useState, useEffect, useRef } from "react";
import { Sidebar } from "../components/Sidebar";
import { ChatInput } from "../components/ChatInput";
import MarkdownRenderer from "../components/MarkdownRenderer";
import { watsonxService } from "../services/watsonxService";

const makeId = () => `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;

export default function App() {
  const [isOpen, setIsOpen] = useState(true);

  const [sessions, setSessions] = useState([{ id: "default", title: "새로운 대화" }]);
  const [activeSessionId, setActiveSessionId] = useState("default");

  const [messagesBySession, setMessagesBySession] = useState({
    default: [],
  });

  const activeMessages = useMemo(
    () => messagesBySession[activeSessionId] || [],
    [messagesBySession, activeSessionId]
  );

  const [disabled, setDisabled] = useState(false);
  
  // 스크롤 컨테이너 ref
  const messagesEndRef = useRef(null);
  const scrollContainerRef = useRef(null);

  // 메시지가 변경될 때마다 최하단으로 스크롤
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [activeMessages]);

  const setMessagesForSession = (sessionId, updater) => {
    setMessagesBySession((prev) => {
      const current = prev[sessionId] || [];
      const next = typeof updater === "function" ? updater(current) : updater;
      return { ...prev, [sessionId]: next };
    });
  };

  const onNewChat = () => {
    const id = makeId();
    const title = "새로운 대화";

    setDisabled(false);
    setSessions((prev) => [{ id, title }, ...prev]);
    setMessagesBySession((prev) => ({ ...prev, [id]: [] }));
    setActiveSessionId(id);

    if (watsonxService?.createNewChat) {
      try {
        watsonxService.createNewChat();
      } catch (e) {
        console.error(e);
      }
    }
  };

  const onSelectSession = (id) => {
    setActiveSessionId(id);
  };

  const onDeleteSession = (id, e) => {
    e.preventDefault();
    e.stopPropagation();

    setSessions((prev) => prev.filter((s) => s.id !== id));
    setMessagesBySession((prev) => {
      const copy = { ...prev };
      delete copy[id];
      return copy;
    });

    if (activeSessionId === id) {
      setActiveSessionId("default");
      setMessagesBySession((prev) => (prev.default ? prev : { ...prev, default: [] }));
      setSessions((prev) =>
        prev.some((s) => s.id === "default") ? prev : [{ id: "default", title: "새로운 대화" }, ...prev]
      );
    }
  };

  const onRenameSession = (id, newTitle) => {
    const t = (newTitle || "").trim();
    if (!t) return;

    setSessions((prev) => prev.map((s) => (s.id === id ? { ...s, title: t } : s)));
  };

  // 가짜 스트리밍 효과 함수
  const simulateTyping = async (sessionId, messageTs, fullText) => {
    console.log('🎬 [타이핑 시작] 총 글자 수:', fullText.length);
    
    let accumulated = "";
    const chars = fullText.split('');
    
    // 로딩 해제
    setMessagesForSession(sessionId, (prev) =>
      prev.map((m) => (m.ts === messageTs ? { ...m, isLoading: false } : m))
    );

    // 글자 단위로 추가
    for (let i = 0; i < chars.length; i++) {
      accumulated += chars[i];
      
      setMessagesForSession(sessionId, (prev) =>
        prev.map((m) => (m.ts === messageTs ? { ...m, content: accumulated } : m))
      );
      
      // 진행률 로그 (10% 단위)
      if (i % Math.floor(chars.length / 10) === 0) {
        console.log(`⏳ [타이핑 진행] ${Math.floor((i / chars.length) * 100)}%`);
      }
      
      // 딜레이 (5ms - 빠른 타이핑)
      if (chars[i] !== '\n' && chars[i] !== ' ') {
        await new Promise(resolve => setTimeout(resolve, 5));
      }
    }
    
    console.log('✅ [타이핑 완료]');
  };

  const onSendMessage = async (text) => {
    if (disabled) return;
    const msg = text?.trim();
    if (!msg) return;

    const sid = activeSessionId;
    
    // 내용증명서 요청인 경우 대화 이력 확인
    if (msg.includes('내용증명서') || msg.includes('문서')) {
      const currentMessages = messagesBySession[sid] || [];
      // 사용자 메시지만 카운트 (assistant 제외)
      const userMessages = currentMessages.filter(m => m.role === 'user');
      
      console.log('📋 [대화 이력 확인] 총 사용자 메시지:', userMessages.length);
      
      if (userMessages.length === 0) {
        // 대화 이력이 없으면 안내 메시지 표시
        const userMsg = { role: "user", content: msg, ts: Date.now() + Math.random() };
        setMessagesForSession(sid, (prev) => [...prev, userMsg]);
        
        const assistantTs = Date.now() + Math.random();
        const warningMsg = "아직 상담 내용이 없습니다. 먼저 의료사고와 관련된 상황을 말씀해 주시면, 그 내용을 바탕으로 내용증명서를 작성해 드리겠습니다. 궁금한 점이 있으시면 지금 말씀해 주세요!";
        
        setMessagesForSession(sid, (prev) => [
          ...prev,
          { role: "assistant", content: warningMsg, ts: assistantTs, isLoading: false }
        ]);
        
        console.log('⚠️ [내용증명서] 대화 이력 없음 - 안내 메시지 표시');
        return;
      }
    }
    
    setDisabled(true);

    const userMsg = { role: "user", content: msg, ts: Date.now() + Math.random() };
    setMessagesForSession(sid, (prev) => [...prev, userMsg]);

    const assistantTs = Date.now() + Math.random();
    setMessagesForSession(sid, (prev) => [
      ...prev,
      { role: "assistant", content: "", ts: assistantTs, isLoading: true }
    ]);

    console.log('🚀 [메시지 전송] 질문:', msg.substring(0, 50) + '...');

    try {
      console.log('⏳ [백엔드 호출 중...]');
      
      // 백엔드 호출 (통합 엔드포인트)
      const response = await watsonxService.sendMessage(msg);

      console.log('📥 [백엔드 응답 수신] 전체 응답:', JSON.stringify(response, null, 2));
      console.log('📥 [백엔드 응답] 타입:', response.type);
      console.log('📥 [백엔드 응답] answer:', response.answer ? '있음' : 'undefined');

      // 응답 타입별 처리
      if (response.type === "document") {
        console.log('📄 [문서 생성 모드]');
        // 문서 작성 응답
        const answer = response.answer || "문서를 작성했습니다.";
        const docContent = response.document_content || "";
        const fullContent = `${answer}\n\n---\n\n${docContent}`;
        console.log('📝 [문서 내용 길이]:', fullContent.length, '자');
        await simulateTyping(sid, assistantTs, fullContent);
        
      } else if (response.type === "chat") {
        console.log('💬 [일반 상담 모드]');
        
        // answer가 없으면 에러
        if (!response.answer) {
          console.error('❌ [에러] response.answer가 undefined입니다!');
          console.error('❌ [전체 응답]:', response);
          setMessagesForSession(sid, (prev) =>
            prev.map((m) =>
              m.ts === assistantTs
                ? { ...m, content: "백엔드에서 응답을 받지 못했습니다. 다시 시도해주세요.", isLoading: false }
                : m
            )
          );
          return;
        }
        
        console.log('💬 [답변 길이]:', response.answer.length, '자');
        console.log('💬 [답변 내용 미리보기]:', response.answer.substring(0, 100) + '...');
        
        // 일반 상담 응답
        await simulateTyping(sid, assistantTs, response.answer);
        
        // TODO: sources 출처 카드 표시 (나중에 구현)
        
      } else if (response.type === "error") {
        console.error('❌ [에러 응답]:', response.error);
        // 에러 처리
        const errorMsg = response.answer || response.error || "알 수 없는 오류가 발생했습니다.";
        setMessagesForSession(sid, (prev) =>
          prev.map((m) =>
            m.ts === assistantTs
              ? { ...m, content: errorMsg, isLoading: false }
              : m
          )
        );
      }

      // 세션 제목 자동 업데이트
      setSessions((prev) =>
        prev.map((s) =>
          s.id === sid && s.title === "새로운 대화"
            ? { ...s, title: (msg || "새로운 대화").slice(0, 18) }
            : s
        )
      );

    } catch (err) {
      console.error('💥 [치명적 에러]:', err);
      setMessagesForSession(sid, (prev) =>
        prev.map((m) =>
          m.ts === assistantTs
            ? {
                ...m,
                content: "오류가 발생했습니다. 백엔드 서버를 확인해주세요.",
                isLoading: false,
              }
            : m
        )
      );
    } finally {
      setDisabled(false);
      console.log('🏁 [메시지 처리 완료]\n');
    }
  };

  const isHome = activeMessages.length === 0;

  const quickPrompts = [
    { icon: "🩺", title: "이 증상, 의료사고일 수 있나요?" },
    { icon: "⚖️", title: "내 사례와 비슷한 의료분쟁 판례 찾아줘" },
    { icon: "🔍", title: "'설명의무 위반'이 무슨 뜻인지 쉽게 알려줘" },
    { icon: "🗂️", title: "의료분쟁 조정 신청 전에 뭘 준비해야 하나요?" },
  ];

  // 체크리스트 정적 데이터
  const checklistData = {
    evidence: `**[필수 증거 체크리스트]**

✅ **진료기록부 사본 (전체)**: 초진 기록, 수술/시술 기록, 경과 기록, 간호 기록지 등 "빠짐없이 전체"를 요청하세요. (의료법상 병원은 거부할 수 없음)

✅ **영상 자료 (CD/USB)**: CT, MRI, X-ray 등 촬영된 모든 영상 자료.

✅ **수술/시술 동의서**: 부작용 설명을 들었다는 서명이 있는 문서.

✅ **결제 영수증**: 진료비 상세 내역서 포함.

✅ **CCTV 영상**: (수술실 내 CCTV가 있는 경우) 보존 기간이 지나기 전에 빠르게 '증거보전신청'을 하거나 열람 요청을 해야 함.

✅ **녹취/사진**: 의료진과의 대화 내용 녹음, 환부의 날짜별 사진.`,

    procedure: `**[한국의료분쟁조정중재원 신청 절차]**

1️⃣ **상담 신청**
전화(1670-2545) 또는 홈페이지 온라인 상담.

2️⃣ **조정 신청서 제출**
사건 개요와 피해 내용을 적어 우편/방문/온라인 제출. (수수료 약 2~3만원 발생)

3️⃣ **피신청인 동의 확인**
병원 측이 조정에 동의해야 절차가 개시됨. (단, 사망/1개월 이상 의식불명/장애등급 1급 등은 자동 개시)

4️⃣ **감정 및 조정**
의료 전문가와 법률가가 과실 유무 판단.

5️⃣ **합의/결정**
조정안을 양측이 받아들이면 재판상 화해와 동일한 효력.`,
  };

  // 체크리스트 버튼 클릭 핸들러 (스트리밍 방식)
  const onChecklistClick = async (type) => {
    if (disabled) return;

    const sid = activeSessionId;
    const buttonTitle = type === 'evidence' ? '필수 증거 체크리스트' : '한국의료분쟁조정중재원 신청 절차';
    const content = checklistData[type];

    setDisabled(true);

    // 사용자 메시지 추가
    const userMsg = { role: "user", content: buttonTitle, ts: Date.now() + Math.random() };
    setMessagesForSession(sid, (prev) => [...prev, userMsg]);

    // 로딩 상태의 빈 assistant 메시지 추가
    const assistantTs = Date.now() + Math.random();
    setMessagesForSession(sid, (prev) => [...prev, { role: "assistant", content: "", ts: assistantTs, isLoading: true }]);

    // 0.5초 지연 (로딩 효과)
    await new Promise(resolve => setTimeout(resolve, 500));

    // 스트리밍 효과로 텍스트 표시
    await simulateTyping(sid, assistantTs, content);

    // 세션 제목 자동 업데이트
    setSessions((prev) =>
      prev.map((s) =>
        s.id === sid && s.title === "새로운 대화"
          ? { ...s, title: buttonTitle.slice(0, 18) }
          : s
      )
    );

    setDisabled(false);
  };

  return (
    <div className="min-h-screen bg-slate-50 flex">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onNewChat={onNewChat}
        onSelectSession={onSelectSession}
        onDeleteSession={onDeleteSession}
        onRenameSession={onRenameSession}
        isOpen={isOpen}
        onToggle={() => setIsOpen((v) => !v)}
      />

      <div className="flex-1 flex flex-col h-screen overflow-hidden">
        <div className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-6 flex-shrink-0">
          <div className="flex flex-col">
            <span className="text-sm font-bold text-slate-900">
              {sessions.find((s) => s.id === activeSessionId)?.title || "새로운 대화"}
            </span>
            <span className="text-[10px] font-bold text-emerald-500 uppercase tracking-wider">
              ● watsonx ACTIVE
            </span>
          </div>

          <div className="flex items-center gap-2">
            <div className="text-[11px] font-bold text-slate-500 bg-slate-100 px-3 py-1.5 rounded-full">
              Pro Version
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto" ref={scrollContainerRef}>
          {isHome ? (
            <div className="px-4 py-10">
              <div className="mx-auto max-w-4xl">
                <div className="text-center">
                  <div className="mx-auto w-16 h-16 rounded-2xl bg-indigo-600/10 flex items-center justify-center mb-6">
                    <div className="w-10 h-10 rounded-xl bg-indigo-600 flex items-center justify-center text-white text-2xl">
                      ✨
                    </div>
                  </div>

                  <h1 className="text-4xl font-black text-slate-900 tracking-tight">
                    무엇을 도와드릴까요?
                  </h1>
                  <p className="mt-3 text-slate-500 font-medium">
                    안녕하세요. <br />
                    지금 겪고 계신 의료 상황을 간단히 말씀해 주세요. <br />
                    제가 몇 가지 질문을 통해 사례를 정리해 드리겠습니다.
                  </p>
                </div>

                <div className="mt-10 grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {quickPrompts.map((p) => (
                    <button
                      key={p.title}
                      onClick={() => onSendMessage(p.title)}
                      className="bg-white border border-slate-200 rounded-2xl p-5 text-left hover:shadow-md transition-all"
                      type="button"
                      disabled={disabled}
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-slate-100 flex items-center justify-center text-xl">
                          {p.icon}
                        </div>
                        <div className="font-bold text-slate-800">{p.title}</div>
                      </div>
                    </button>
                  ))}
                </div>

                {/* 체크리스트 버튼 - 홈 화면에만 표시 */}
                <div className="mt-8 flex justify-center gap-3 flex-wrap">
                  <button
                    onClick={() => onChecklistClick('evidence')}
                    className="text-xs font-semibold text-slate-700 bg-white border border-slate-300 px-4 py-2 rounded-full hover:bg-slate-50 hover:border-slate-400 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    type="button"
                    disabled={disabled}
                  >
                    📋 필수 증거 체크리스트
                  </button>
                  <button
                    onClick={() => onChecklistClick('procedure')}
                    className="text-xs font-semibold text-slate-700 bg-white border border-slate-300 px-4 py-2 rounded-full hover:bg-slate-50 hover:border-slate-400 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    type="button"
                    disabled={disabled}
                  >
                    ⚖️ 한국의료분쟁조정중재원 신청 절차
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="px-4 py-6">
              <div className="mx-auto max-w-4xl space-y-4">
                {activeMessages.map((m) => (
                  <div
                    key={m.ts}
                    className={`rounded-2xl p-4 ${
                      m.role === "user"
                        ? "bg-indigo-600 text-white ml-auto max-w-[80%]"
                        : "bg-white border border-slate-200 max-w-[80%]"
                    }`}
                  >
                    {m.role === "assistant" ? (
                      m.isLoading ? (
                        <div className="flex items-center gap-1 text-slate-400">
                          <span className="animate-bounce" style={{ animationDelay: '0ms', animationDuration: '1s' }}>.</span>
                          <span className="animate-bounce" style={{ animationDelay: '200ms', animationDuration: '1s' }}>.</span>
                          <span className="animate-bounce" style={{ animationDelay: '400ms', animationDuration: '1s' }}>.</span>
                        </div>
                      ) : (
                        <MarkdownRenderer content={m.content} />
                      )
                    ) : (
                      <div className="font-medium">{m.content}</div>
                    )}
                  </div>
                ))}
                {/* 스크롤 타겟 마커 */}
                <div ref={messagesEndRef} />
              </div>
            </div>
          )}
        </div>

        <div className="flex-shrink-0 relative">
          {/* 내용증명서 작성 플로팅 버튼 - 채팅 시작 후에만 표시 */}
          {!isHome && (
            <div className="absolute -top-16 right-6 group">
              <button
                onClick={() => onSendMessage('지금까지 상담한 내용을 바탕으로 의료사고 손해배상 청구 내용증명서를 작성해줘')}
                className="w-12 h-12 rounded-full bg-indigo-600 text-white shadow-lg hover:bg-indigo-700 hover:shadow-xl transition-all flex items-center justify-center text-xl disabled:opacity-50 disabled:cursor-not-allowed"
                type="button"
                disabled={disabled}
                aria-label="내용증명서 작성"
              >
                📄
              </button>
              
              {/* 툴팁 말풍선 */}
              <div className="absolute bottom-full right-0 mb-2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                <div className="bg-slate-800 text-white text-xs font-medium px-3 py-2 rounded-lg whitespace-nowrap shadow-lg">
                  지금 내용으로 내용증명서 작성
                  {/* 말풍선 꼬리 */}
                  <div className="absolute top-full right-4 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-slate-800"></div>
                </div>
              </div>
            </div>
          )}
          
          <ChatInput onSendMessage={onSendMessage} disabled={disabled} />
        </div>
      </div>
    </div>
  );
}