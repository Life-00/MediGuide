import React, { useEffect, useMemo, useState } from "react";
import { Sidebar } from "../components/Sidebar";
import { ChatInput } from "../components/ChatInput";
import MarkdownRenderer from "../components/MarkdownRenderer";
import { geminiService } from "../services/geminiService";

// 메시지 형태: { role: "user" | "assistant", content: string, ts: number }
const now = () => Date.now();

export default function App() {
  const [isOpen, setIsOpen] = useState(true);

  // 세션 목록
  const [sessions, setSessions] = useState([{ id: "default", title: "새로운 대화" }]);
  const [activeSessionId, setActiveSessionId] = useState("default");

  // 세션별 메시지 저장(메모리)
  const [messagesBySession, setMessagesBySession] = useState({
    default: [],
  });

  const activeMessages = useMemo(
    () => messagesBySession[activeSessionId] || [],
    [messagesBySession, activeSessionId]
  );

  const [disabled, setDisabled] = useState(false);

  // 세션 바뀔 때마다 새 채팅(원하시면 제거 가능)
  useEffect(() => {
    if (geminiService?.createNewChat) {
      geminiService.createNewChat();
    }
  }, [activeSessionId]);

  const setActiveMessages = (updater) => {
    setMessagesBySession((prev) => {
      const current = prev[activeSessionId] || [];
      const next = typeof updater === "function" ? updater(current) : updater;
      return { ...prev, [activeSessionId]: next };
    });
  };

  const onNewChat = () => {
    const id = `session-${now()}`;
    const title = "새로운 대화";
    setSessions((prev) => [{ id, title }, ...prev]);
    setMessagesBySession((prev) => ({ ...prev, [id]: [] }));
    setActiveSessionId(id);
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
    }
  };

  const onSendMessage = async (text) => {
    const userMsg = { role: "user", content: text, ts: now() };
    setActiveMessages((prev) => [...prev, userMsg]);

    // assistant placeholder
    const assistantTs = now();
    setActiveMessages((prev) => [...prev, { role: "assistant", content: "", ts: assistantTs }]);

    setDisabled(true);

    try {
      // 스트리밍 응답 누적
      let acc = "";
      if (!geminiService?.sendMessageStream) {
        throw new Error("geminiService.sendMessageStream not found");
      }

      for await (const chunk of geminiService.sendMessageStream(text)) {
        acc += chunk || "";
        setActiveMessages((prev) =>
          prev.map((m) => (m.ts === assistantTs ? { ...m, content: acc } : m))
        );
      }

      // 사이드바 타이틀 자동 갱신 (처음 질문으로)
      setSessions((prev) =>
        prev.map((s) =>
          s.id === activeSessionId && s.title === "새로운 대화"
            ? { ...s, title: (text || "새로운 대화").slice(0, 18) }
            : s
        )
      );
    } catch (err) {
      setActiveMessages((prev) =>
        prev.map((m) =>
          m.ts === assistantTs
            ? {
                ...m,
                content:
                  "오류가 발생했습니다. (API Key/패키지 설치/브라우저 환경) 설정을 확인해주세요.",
              }
            : m
        )
      );
      // 콘솔 로그
      console.error(err);
    } finally {
      setDisabled(false);
    }
  };

  // “홈페이지처럼” 보이는 상태: 메시지가 0개일 때
  const isHome = activeMessages.length === 0;

  const quickPrompts = [
    { icon: "🩺", title: "이 증상, 의료사고일 수 있나요?" },
    { icon: "⚖️", title: "내 사례와 비슷한 의료분쟁 판례 찾아줘" },
    { icon: "🔍", title: "‘설명의무 위반’이 무슨 뜻인지 쉽게 알려줘" },
    { icon: "🗂️", title: "의료분쟁 조정 신청 전에 뭘 준비해야 하나요?" },
  ];

  const handleQuickPrompt = (t) => onSendMessage(t);

  return (
    <div className="min-h-screen bg-slate-50 flex">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onNewChat={onNewChat}
        onSelectSession={onSelectSession}
        onDeleteSession={onDeleteSession}
        isOpen={isOpen}
        onToggle={() => setIsOpen((v) => !v)}
      />

      {/* Main */}
      <div className="flex-1 flex flex-col min-h-screen">
        {/* Top bar (홈페이지 느낌) */}
        <div className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-6">
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

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          {isHome ? (
            // 홈 화면(추천 카드)
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
                    안녕하세요.  
                    지금 겪고 계신 의료 상황을 간단히 말씀해 주세요.  
                    제가 몇 가지 질문을 통해 사례를 정리해 드리겠습니다.
                  </p>
                </div>

                <div className="mt-10 grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {quickPrompts.map((p) => (
                    <button
                      key={p.title}
                      onClick={() => handleQuickPrompt(p.title)}
                      className="bg-white border border-slate-200 rounded-2xl p-5 text-left hover:shadow-md transition-all"
                      type="button"
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
              </div>
            </div>
          ) : (
            // 채팅 화면
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
                      <MarkdownRenderer content={m.content} />
                    ) : (
                      <div className="font-medium">{m.content}</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Input */}
        <ChatInput onSendMessage={onSendMessage} disabled={disabled} />
      </div>
    </div>
  );
}
