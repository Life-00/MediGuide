import React, { useState, useRef, useEffect } from "react";
import { Send, Paperclip, Zap } from "lucide-react";

export const ChatInput = ({ onSendMessage, disabled = false }) => {
  const [input, setInput] = useState("");
  const textareaRef = useRef(null);

  const handleSend = () => {
    if (input.trim() && !disabled) {
      onSendMessage(input.trim());
      setInput("");
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        200
      )}px`;
    }
  }, [input]);

  return (
    <div className="bg-white px-4 pb-6 pt-2">
      <div className="mx-auto max-w-4xl">
        <div
          className={`
            relative flex items-end gap-3 bg-white rounded-[24px] border-2 p-3 transition-all duration-300
            ${
              disabled
                ? "border-slate-100 bg-slate-50"
                : "border-slate-200 shadow-lg shadow-slate-200/50 focus-within:border-indigo-500 focus-within:shadow-indigo-100"
            }
          `}
        >
          <button
            className="p-2.5 text-slate-400 hover:text-indigo-500 hover:bg-indigo-50 rounded-full transition-all"
            title="Attach file"
            type="button"
          >
            <Paperclip size={20} />
          </button>

          <textarea
            ref={textareaRef}
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="궁금한 것을 물어보세요..."
            className="flex-1 bg-transparent border-none focus:ring-0 resize-none py-2.5 px-1 text-slate-700 placeholder:text-slate-400 font-medium max-h-[200px]"
            disabled={disabled}
          />

          <button
            onClick={handleSend}
            disabled={!input.trim() || disabled}
            className={`
              group flex items-center justify-center p-3 rounded-2xl transition-all duration-300
              ${
                input.trim() && !disabled
                  ? "bg-indigo-600 text-white hover:bg-indigo-700 shadow-lg shadow-indigo-200 active:scale-95"
                  : "bg-slate-100 text-slate-300"
              }
            `}
            type="button"
          >
            {disabled ? (
              <div className="w-5 h-5 border-2 border-slate-300 border-t-white rounded-full animate-spin" />
            ) : (
              <Send
                size={20}
                className={
                  input.trim()
                    ? "translate-x-0.5 -translate-y-0.5 rotate-[-10deg] transition-transform"
                    : ""
                }
              />
            )}
          </button>
        </div>

        <div className="flex items-center justify-center gap-4 mt-3">
          <div className="flex items-center gap-1">
            <Zap size={10} className="text-amber-500 fill-amber-500" />
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider italic">
              Ultra Fast Response Mode
            </span>
          </div>
          <span className="text-[10px] text-slate-300">|</span>
          <p className="text-[10px] font-medium text-slate-400">
            Gemini 3 Pro 모델이 최적의 답변을 생성합니다.
          </p>
        </div>
      </div>
    </div>
  );
};
