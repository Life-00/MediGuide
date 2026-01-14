import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
// 아래는 사용 중일 때만 유지하세요.
// import rehypeHighlight from "rehype-highlight";
// import "highlight.js/styles/github.css";

/**
 * content 전처리:
 * 1) 단일 개행도 화면 줄바꿈으로 보이도록 "하드 브레이크(끝 공백 2개 + \n)" 처리
 * 2) "항목 제목" 다음 줄(설명 줄)에 들여쓰기(여백) 적용
 *    - 항목 제목: **☐ ...** 또는 **① ...** 같은 굵은 제목 라인
 *    - HTML 태그 없이, 유니코드 NBSP(\u00A0)로 들여쓰기
 * 3) 코드블록(```) 내부는 원문 그대로 유지
 */
function preprocessMarkdown(input = "") {
  const lines = String(input).replace(/\r\n/g, "\n").split("\n");

  let inCodeBlock = false;
  let afterItemTitle = false;

  const NBSP = "\u00A0";
  const INDENT = NBSP.repeat(4); // 들여쓰기 폭 (원하면 3~6으로 조절)

  const isSectionTitleLine = (line) => /^#{1,6}\s+/.test(line.trim()); // #, ##, ### ...
  const isEmptyLine = (line) => line.trim() === "";

  // ✅ 항목 제목 라인 판별:
  // - **☐ ...**
  // - **① ...** (①~⑳ 등 원형 숫자)
  // - 필요 시 확장 가능
  const isItemTitleLine = (line) => {
    const t = line.trim();

    // **...** 형태인지
    const isBoldLine = t.startsWith("**") && t.endsWith("**") && t.length >= 4;
    if (!isBoldLine) return false;

    // 내부 텍스트
    const inner = t.slice(2, -2).trim();

    // 체크박스 시작 또는 원형 숫자 시작(①~⑳)
    const startsWithCheckbox = inner.startsWith("☐");
    const startsWithCircledNumber = /^[①-⑳]/.test(inner);

    return startsWithCheckbox || startsWithCircledNumber;
  };

  const out = [];

  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];

    // 코드블록 토글 (```로 시작/종료)
    if (line.trim().startsWith("```")) {
      inCodeBlock = !inCodeBlock;
      out.push(line);
      continue;
    }

    // 코드블록 내부는 그대로
    if (inCodeBlock) {
      out.push(line);
      continue;
    }

    // 새 섹션 제목/빈 줄/다음 항목 제목이 나오면 "설명 들여쓰기 모드" 해제
    if (isSectionTitleLine(line) || isEmptyLine(line) || isItemTitleLine(line)) {
      afterItemTitle = false;
    }

    // 항목 제목 라인
    if (isItemTitleLine(line)) {
      afterItemTitle = true;
      // 제목 라인도 줄바꿈 유지
      out.push(line + "  ");
      continue;
    }

    // 항목 제목 다음의 설명 줄들에 들여쓰기 적용
    if (afterItemTitle && !isEmptyLine(line) && !isSectionTitleLine(line)) {
      const trimmed = line.replace(/^\s+/, "");
      line = INDENT + trimmed;
    }

    // 단일 개행도 줄바꿈 되도록 하드 브레이크 적용
    out.push(line + "  ");
  }

  return out.join("\n");
}

/**
 * props:
 * - content: string
 */
export default function MarkdownRenderer({ content = "" }) {
  const rendered = preprocessMarkdown(content);

  return (
    <div className="prose prose-slate max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        // rehypePlugins={[rehypeHighlight]} // 사용 중일 때만
        components={{
          // ✅ "### ..." 타이틀: 위 여백 줄이고, 크기 조금 낮춤(굵기는 유지)
          h3: ({ node, ...props }) => (
            <h3
              {...props}
              className="!mt-3 !mb-3 !text-xl !font-extrabold !text-slate-900"
            />
          ),

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
        {rendered}
      </ReactMarkdown>
    </div>
  );
}
