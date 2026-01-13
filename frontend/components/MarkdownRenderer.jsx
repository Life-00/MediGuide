import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
// 아래는 사용 중일 때만 유지하세요.
// import rehypeHighlight from "rehype-highlight";
// import "highlight.js/styles/github.css";

/**
 * props:
 * - content: string
 */
export default function MarkdownRenderer({ content = "" }) {
  return (
    <div className="prose prose-slate max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        // rehypePlugins={[rehypeHighlight]} // 사용 중일 때만
        components={{
          a: ({ node, ...props }) => (
            <a
              {...props}
              target="_blank"
              rel="noreferrer"
              className="text-indigo-600 hover:underline"
            />
          ),
          code: ({ inline, className, children, ...props }) => {
            // ```lang 코드블록을 깔끔한 문서 스타일로 표시
            if (!inline) {
              return (
                <div className="my-4 rounded-lg bg-white border-2 border-slate-200 p-6 shadow-sm">
                  <pre className="whitespace-pre-wrap break-words font-sans text-sm text-slate-800 leading-relaxed">
                    {children}
                  </pre>
                </div>
              );
            }
            // 인라인 코드
            return (
              <code
                className="rounded bg-slate-100 px-1 py-0.5 text-slate-800"
                {...props}
              >
                {children}
              </code>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}