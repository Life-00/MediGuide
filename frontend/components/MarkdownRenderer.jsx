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
            // ```lang 코드블록 지원용
            if (!inline) {
              return (
                <pre className="overflow-auto rounded-xl bg-slate-900 p-4 text-slate-100">
                  <code className={className} {...props}>
                    {children}
                  </code>
                </pre>
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
