# 🏥 MediGuide – 의료분쟁 솔루션 AI 챗봇 (Frontend)

<img width="1920" height="921" alt="image" src="https://github.com/user-attachments/assets/b23b8eff-a32c-4883-bc0b-5d75a6cab285" />
<img width="1920" height="914" alt="image" src="https://github.com/user-attachments/assets/c814e1f4-2a2d-407f-9a20-59ac02cf6a67" />


## 🧱 기술 스택 (Tech Stack)

- **Frontend**: React + Vite
- **Styling**: Tailwind CSS
- **AI API**: Google Gemini
- **Markdown Rendering**: react-markdown, remark-gfm
- **Package Manager**: npm
- **Version Control**: Git / GitHub

---

## 📁 프로젝트 구조

```text
frontend/
├─ assets/                # 이미지, 아이콘 등 정적 리소스
├─ components/             # 공통 UI 컴포넌트
│  ├─ ChatInput.jsx        # 하단 채팅 입력 컴포넌트
│  ├─ MarkdownRenderer.jsx # AI 응답(Markdown) 렌더링
│  └─ Sidebar.jsx          # 좌측 사이드바 (대화 목록/세션 관리)
├─ hooks/                  # (확장 대비) 커스텀 React Hooks
├─ pages/                  # (확장 대비) 페이지 단위 컴포넌트
├─ public/                 # 정적 파일
├─ services/
│  └─ geminiService.js     # Gemini API 연동 및 스트리밍 처리
├─ src/
│  ├─ App.jsx              # 메인 화면 (홈 + 채팅 UI 제어)
│  ├─ main.jsx             # React 엔트리 포인트
│  └─ index.css            # Tailwind CSS 엔트리 파일
├─ styles/                 # (확장 대비) 글로벌 스타일
├─ utils/                  # (확장 대비) 공통 유틸 함수
├─ index.html              # HTML 엔트리
├─ tailwind.config.js      # Tailwind 설정
├─ postcss.config.js       # PostCSS 설정
├─ vite.config.js          # Vite 설정
├─ package.json            # npm 의존성 정의
└─ README.md               # 프로젝트 설명 문서
````

---

## ⚙️ 필수 실행 환경

* **Node.js**: v18 이상 권장
* **npm**: v9 이상 권장

---

## 📦 설치 방법

```bash
# 1. 레포지토리 클론
git clone <repository-url>
cd MediGuide/frontend

# 2. 의존성 설치
npm install
```

---

## 🔐 환경변수 설정

`frontend/.env.local` 파일 생성 후 아래 내용 추가:

```env
VITE_API_KEY=YOUR_GEMINI_API_KEY
```

> ⚠️ 해당 파일은 `.gitignore`에 포함되어야 하며,
> API Key는 외부에 노출되면 안 됩니다.

---

## ▶️ 실행 방법

```bash
npm run dev
```

* 브라우저 접속:
  👉 [http://localhost:5173](http://localhost:5173)

---

## 🖥️ 주요 화면 설명

* **홈 화면**

  * 의료분쟁 관련 Quick Prompt 제공
  * 사용자가 바로 질문을 시작할 수 있는 진입점

* **채팅 화면**

  * Gemini 기반 AI 응답 스트리밍
  * Markdown 기반 구조화된 답변 제공

* **사이드바**

  * 대화 세션 생성 / 선택 / 삭제
  * 최근 대화 목록 관리

---

## ⚠️ 법적 고지 (Disclaimer)

본 서비스는 **법률 자문을 제공하지 않으며**,
의료분쟁 조정·중재 사례 기반의 **정보 제공**을 목적으로 합니다.

---

## 👥 팀원 참고 사항

* 현재 프론트엔드는 **Gemini API를 직접 호출하는 구조**
* 실서비스 단계에서는 **백엔드 서버를 통한 AI 호출 구조로 분리 예정**
* Tailwind CSS 기반 UI이므로 스타일 관련 수정 시 Tailwind 클래스 기준으로 작업
