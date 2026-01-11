import React, { useEffect, useState } from "react";
import {
  Plus,
  MessageSquare,
  Settings,
  PanelLeftClose,
  Trash2,
  Search,
  Pencil,
} from "lucide-react";

export const Sidebar = ({
  sessions = [],
  activeSessionId = null,
  onNewChat,
  onSelectSession,
  onDeleteSession,
  onRenameSession, // ✅ 추가
  isOpen,
  onToggle,
}) => {
  const [editingId, setEditingId] = useState(null);
  const [draftTitle, setDraftTitle] = useState("");

  // 활성 세션이 바뀌면 편집 상태 해제(원치 않으면 제거 가능)
  useEffect(() => {
    setEditingId(null);
    setDraftTitle("");
  }, [activeSessionId]);

  const startEdit = (session) => {
    setEditingId(session.id);
    setDraftTitle(session.title || "");
  };

  const commitEdit = (sessionId) => {
    const t = (draftTitle || "").trim();
    if (t) onRenameSession?.(sessionId, t);
    setEditingId(null);
    setDraftTitle("");
  };

  const cancelEdit = () => {
    setEditingId(null);
    setDraftTitle("");
  };

  return (
    <div
      className={`
        fixed inset-y-0 left-0 z-50 w-80 bg-slate-900 text-slate-300 transition-all duration-300 ease-in-out lg:relative lg:translate-x-0
        ${isOpen ? "translate-x-0" : "-translate-x-full"}
      `}
    >
      <div className="flex h-full flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 mb-2">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-indigo-700 flex items-center justify-center text-white shadow-lg border border-indigo-400/20">
              <Plus size={24} strokeWidth={3} />
            </div>
            <div className="flex flex-col">
              <span className="font-bold text-lg text-white tracking-tight leading-none">
                Concierge
              </span>
              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mt-1">
                Enterprise AI
              </span>
            </div>
          </div>
          <button
            onClick={onToggle}
            className="p-2 hover:bg-slate-800 rounded-xl lg:hidden text-slate-500 hover:text-white transition-colors"
            type="button"
          >
            <PanelLeftClose size={20} />
          </button>
        </div>

        {/* Action Button */}
        <div className="px-6 mb-8">
          <button
            onClick={onNewChat}
            className="flex w-full items-center justify-center gap-2 rounded-2xl bg-indigo-600 px-4 py-4 text-sm font-bold text-white transition-all hover:bg-indigo-500 shadow-xl shadow-indigo-900/20 active:scale-[0.98]"
            type="button"
          >
            <Plus size={18} />
            새로운 프로젝트 시작
          </button>
        </div>

        {/* Search Mockup */}
        <div className="px-6 mb-4 relative">
          <Search
            className="absolute left-9 top-1/2 -translate-y-1/2 text-slate-500"
            size={14}
          />
          <input
            type="text"
            placeholder="대화 검색..."
            className="w-full bg-slate-800/50 border border-slate-700 rounded-xl py-2 pl-9 pr-4 text-xs focus:ring-1 focus:ring-indigo-500 outline-none transition-all"
          />
        </div>

        {/* Chat History List */}
        <div className="flex-1 overflow-y-auto px-4 space-y-2 custom-scrollbar">
          <div className="flex items-center justify-between px-2 mb-3">
            <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">
              최근 대화 목록
            </h3>
            <span className="text-[10px] font-bold text-slate-600 bg-slate-800 px-2 py-0.5 rounded-full">
              {sessions.length}
            </span>
          </div>

          {sessions.length === 0 ? (
            <div className="px-3 py-10 text-center">
              <div className="text-slate-700 mb-2 italic text-sm">
                기록이 없습니다
              </div>
            </div>
          ) : (
            sessions.map((session) => {
              const isActive = activeSessionId === session.id;
              const isEditing = editingId === session.id;

              return (
                <button
                  key={session.id}
                  onClick={() => onSelectSession(session.id)}
                  className={`group flex w-full items-center gap-3 rounded-xl px-4 py-3.5 text-sm transition-all relative ${
                    isActive
                      ? "bg-slate-800 text-white shadow-inner border-l-4 border-indigo-500"
                      : "hover:bg-slate-800/50 text-slate-400 hover:text-slate-200"
                  }`}
                  type="button"
                >
                  <MessageSquare
                    size={16}
                    className={isActive ? "text-indigo-400" : "text-slate-600"}
                  />

                  {/* 제목 영역: 보기/편집 전환 */}
                  <div className="flex-1 min-w-0">
                    {isEditing ? (
                      <input
                        value={draftTitle}
                        onChange={(e) => setDraftTitle(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            e.preventDefault();
                            commitEdit(session.id);
                          } else if (e.key === "Escape") {
                            e.preventDefault();
                            cancelEdit();
                          }
                        }}
                        onBlur={() => commitEdit(session.id)}
                        autoFocus
                        className="w-full bg-slate-900/40 border border-slate-600 rounded-lg px-2 py-1 text-sm text-slate-100 outline-none"
                      />
                    ) : (
                      <span className="block truncate text-left font-medium">
                        {session.title}
                      </span>
                    )}
                  </div>

                  {/* 편집 아이콘 */}
                  {!isEditing && (
                    <div
                      onClick={(e) => {
                        e.stopPropagation();
                        startEdit(session);
                      }}
                      className={`p-1.5 rounded-md hover:bg-slate-700/60 hover:text-slate-100 transition-all ${
                        isActive ? "opacity-100" : "opacity-0 group-hover:opacity-100"
                      }`}
                      role="button"
                      tabIndex={0}
                      title="이름 변경"
                    >
                      <Pencil size={14} />
                    </div>
                  )}

                  {/* 삭제 아이콘 */}
                  <div
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteSession(session.id, e);
                    }}
                    className={`p-1.5 rounded-md hover:bg-red-500/20 hover:text-red-400 transition-all ${
                      isActive ? "opacity-100" : "opacity-0 group-hover:opacity-100"
                    }`}
                    role="button"
                    tabIndex={0}
                    title="삭제"
                  >
                    <Trash2 size={14} />
                  </div>
                </button>
              );
            })
          )}
        </div>

        {/* Footer */}
        <div className="mt-auto border-t border-slate-800/50 p-6 bg-slate-900/50">
          <div className="flex flex-col gap-2">
            <button
              className="flex w-full items-center gap-3 rounded-xl px-4 py-2.5 text-sm font-semibold text-slate-400 transition-colors hover:bg-slate-800 hover:text-white"
              type="button"
            >
              <Settings size={18} />
              설정
            </button>
          </div>

          <div className="mt-6 flex items-center gap-4 px-2">
            <div className="h-10 w-10 rounded-2xl bg-gradient-to-tr from-slate-700 to-slate-800 border border-slate-600 flex items-center justify-center text-slate-400 font-bold">
              G
            </div>
            <div className="flex flex-col overflow-hidden">
              <span className="text-sm font-bold text-white truncate">
                Guest User
              </span>
              <span className="text-[10px] font-bold text-indigo-400 uppercase tracking-tighter">
                Premium Account
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};