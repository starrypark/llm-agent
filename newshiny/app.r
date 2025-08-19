# app.R
library(shiny)
library(httr2)
library(jsonlite)
library(htmltools)
library(bslib)

# ----------------- 설정 -----------------
backend_url <- "http://127.0.0.1:8000/chat"
backend_url_stream <- "http://127.0.0.1:8000/chat_stream"
backend_url_reset <- "http://127.0.0.1:8000/reset_memory"

safe_post_fixed <- function(url, body, timeout_sec = 60){
  req <- httr2::request(url) |>
    httr2::req_method("POST") |>
    httr2::req_body_json(body) |>
    httr2::req_timeout(timeout_sec)
  tryCatch({
    resp <- httr2::req_perform(req)
    list(ok = TRUE, data = jsonlite::fromJSON(httr2::resp_body_string(resp)))
  }, error = function(e){
    list(ok = FALSE, msg = conditionMessage(e))
  })
}
`%||%` <- function(x, y) if (is.null(x)) y else x

# ----------------- COOL-TONE THEME -----------------
theme_cool <- bs_theme(
  version = 5,
  bootswatch = "darkly",
  primary = "#60a5fa",   # cool blue
  secondary = "#22d3ee", # cyan
  success = "#34d399",
  info = "#67e8f9",
  warning = "#fbbf24",
  danger = "#f43f5e",
  base_font = font_google("Inter"),
  heading_font = font_google("Poppins"),
  code_font = font_google("JetBrains Mono")
)

# ----------------- UI -----------------
ui_core <- page_navbar(
  title = tags$div(
    class="d-flex align-items-center gap-2",
    tags$span(class="bi bi-snow"),
    "AI Bot Example"
  ),
  theme = theme_cool,
  window_title = "Shiny + FastAPI — Cool Tone",
  
  # ---- 메인 탭 ----
  nav_panel(
    "Chat",
    layout_column_wrap(
      width = 1,
      card(
        full_screen = FALSE,
        class = "glass-card",
        style = "max-width: 1100px; margin: 24px auto;",
        card_header(
          tags$div(
            class="d-flex align-items-center justify-content-between",
            tags$div(
              tags$h4(class="m-0", "AI Agent Bot Test"),
              tags$small(class="text-muted", "OpenAI 호출 · Memory · Streaming")
            ),
            tags$div(
              class="d-flex align-items-center gap-2",
              tags$a(href="https://fastapi.tiangolo.com/", target="_blank",
                     class="btn btn-sm btn-outline-info",
                     tags$span(class="bi bi-box-arrow-up-right"), " FastAPI"),
              tags$a(href="https://shiny.posit.co/", target="_blank",
                     class="btn btn-sm btn-outline-info",
                     tags$span(class="bi bi-box-arrow-up-right"), " Shiny")
            )
          )
        ),
        card_body(
          # ---- 스타일: 배경/버블/글래스모피즘/네온 ----
          tags$style(HTML("
            :root{
              --bg1: #0b1220;  /* deep navy */
              --bg2: #0f172a;  /* slate-900 */
              --glass: rgba(15, 23, 42, 0.35);
              --border: rgba(148,163,184,0.15);
              --user: linear-gradient(135deg,#22d3ee 0%, #60a5fa 100%);
              --bot:  linear-gradient(135deg,#1f2937 0%, #334155 100%);
              --glow: 0 0 20px rgba(96,165,250,0.25);
              --accent: #67e8f9;
            }

            /* 배경: 크로스 그라데이션 + 블러 블롭 */
            body {
              background: radial-gradient(60% 80% at 20% 10%, rgba(34,211,238,0.12) 0%, rgba(34,211,238,0) 55%),
                          radial-gradient(70% 90% at 80% 20%, rgba(96,165,250,0.10) 0%, rgba(96,165,250,0) 60%),
                          linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 100%) !important;
              background-attachment: fixed !important;
            }

            .glass-card{
              background: var(--glass) !important;
              border: 1px solid var(--border) !important;
              box-shadow: 0 10px 30px rgba(2,8,23,0.5), var(--glow) !important;
              backdrop-filter: blur(10px) saturate(120%);
              -webkit-backdrop-filter: blur(10px) saturate(120%);
              border-radius: 16px;
            }

            .chat-wrap { max-width: 1100px; margin: 0 auto; }
            .chat-box {
              height: 58vh; min-height: 440px;
              border: 1px solid var(--border); border-radius: 14px;
              padding: 14px 14px 6px 14px; overflow-y: auto;
              background: rgba(2, 6, 23, 0.4);
              box-shadow: inset 0 0 50px rgba(99,102,241,0.03);
            }

            .bubble {
              padding: 12px 14px; border-radius: 16px; margin: 8px 0;
              max-width: 82%;
              border: 1px solid var(--border);
              color: #e2e8f0; line-height: 1.45;
              box-shadow: 0 6px 18px rgba(2,8,23,0.35);
            }
            .bubble b { font-weight: 600; color: #e5f4ff; }
            .bubble .meta { font-size: 12px; color: #a1a1aa; margin: 4px 0 8px; }

            .me{
              margin-left: auto;
              border-top-right-radius: 6px;
              background: var(--user);
              color: #0b1220;
              text-shadow: 0 1px 0 rgba(255,255,255,0.25);
            }
            .me b{ color:#0b1220; }
            .bot{
              margin-right: auto;
              border-top-left-radius: 6px;
              background: var(--bot);
            }

            .sticky-input {
              position: sticky; bottom: 0; padding-top: 12px;
              border-top: 1px dashed var(--border);
              background: linear-gradient(180deg, rgba(2,6,23,0) 0%, rgba(2,6,23,0.45) 60%, rgba(2,6,23,0.75) 100%);
            }
            .sid-input { max-width: 360px; }

            .neon-input textarea, .neon-input input {
              background: rgba(2,6,23,0.55) !important;
              border: 1px solid var(--border) !important;
              color: #e2e8f0 !important;
              box-shadow: inset 0 0 0 rgba(0,0,0,0), 0 0 0 rgba(0,0,0,0);
            }
            .neon-input textarea:focus, .neon-input input:focus {
              outline: none !important;
              border-color: var(--accent) !important;
              box-shadow: 0 0 0 2px rgba(103,232,249,0.25), var(--glow) !important;
            }

            .btn-primary {
              background: linear-gradient(135deg,#3b82f6 0%, #22d3ee 100%) !important;
              border: 0 !important; color: #06121f !important; font-weight: 600;
              text-shadow: 0 1px 0 rgba(255,255,255,0.25);
              box-shadow: 0 8px 20px rgba(34,211,238,0.25);
            }
            .btn-outline-primary{
              border-color: #22d3ee !important; color: #a5f3fc !important;
            }
            .btn-outline-primary:hover{
              background: rgba(34,211,238,0.15) !important;
            }

            .spinner {
              display: none; margin-left: 8px;
              width: 18px; height: 18px; border: 2px solid rgba(148,163,184,0.25);
              border-top-color: #60a5fa; border-radius: 50%;
              animation: spin 0.7s linear infinite;
              filter: drop-shadow(0 0 6px rgba(96,165,250,0.35));
            }
            @keyframes spin { to { transform: rotate(360deg); } }
          ")),
          
          # 상단 안내/세션
          tags$div(
            class="d-flex flex-wrap align-items-end justify-content-between gap-3 mb-3",
            div(
              class="sid-input neon-input",
              textInput("sid", label = "session_id", placeholder = "ex) niceguy-001")
            ),
            div(
              class="text-muted",
              tags$small(
                tags$span(class="bi bi-info-circle"),
                "같은 session_id로 보내면 이전 대화를 기억합니다."
              )
            )
          ),
          
          # 채팅 영역
          div(
            class="chat-wrap",
            div(
              class="chat-box",
              htmlOutput("log"),
              tags$div(id="log_stream")
            ),
            div(
              class="sticky-input neon-input",
              textAreaInput("q", NULL, rows = 3, placeholder = "질문을 입력하세요"),
              div(
                class="d-flex align-items-center gap-2 mt-1",
                actionButton("send",
                             label = HTML('<span class="bi bi-send"></span> Send (non-stream)'),
                             class = "btn btn-primary"),
                actionButton("send_stream",
                             label = HTML('<span class="bi bi-lightning"></span> Send (stream)'),
                             class = "btn btn-outline-primary"),
                actionButton("clear_chat",
                             label = HTML('<span class="bi bi-trash"></span> Clear chat'),
                             class = "btn btn-outline-danger"),
                tags$div(id="loadingSpinner", class="spinner")
              )
            )
          )
        )
      )
    ),
    
    # ---- 스트리밍 JS ----
    tags$script(HTML(sprintf("
      (function(){
        const btn = document.getElementById('%s');
        const logStream = document.getElementById('log_stream');
        const spinner = document.getElementById('loadingSpinner');

        function escapeHtml(text){
          var div = document.createElement('div');
          div.innerText = text;
          return div.innerHTML;
        }
        function addBubble(html, who){
          const wrap = document.createElement('div');
          wrap.className = 'bubble ' + (who === 'me' ? 'me' : 'bot');
          wrap.innerHTML = html;
          return wrap;
        }
        function scrollToBottom(){
          const box = document.querySelector('.chat-box');
          if(box) box.scrollTop = box.scrollHeight;
        }

        btn.addEventListener('click', async function(){
          const sid = document.getElementById('%s').value || 'default';
          const qElm = document.getElementById('%s');
          const q = qElm.value.trim();
          if(!q) return;

          // 유저 말풍선
          const you = addBubble('<b>You</b><div class=\"meta\">stream</div><div>'+ escapeHtml(q) +'</div>', 'me');
          logStream.appendChild(you);

          qElm.value = '';
          scrollToBottom();

          // 어시스턴트 말풍선
          const asDiv = addBubble('<b>Assistant</b><div class=\"meta\">streaming...</div><div id=\"streamBody\"></div>', 'bot');
          logStream.appendChild(asDiv);
          const bodyEl = asDiv.querySelector('#streamBody');

          spinner.style.display = 'inline-block';
          try {
            const r = await fetch('%s', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({ question: q, session_id: sid })
            });
            if(!r.ok || !r.body){
              const txt = await r.text();
              bodyEl.innerHTML = '<span style=\"color:#fca5a5\"><b>Error:</b> '+ escapeHtml(txt) +'</span>';
              spinner.style.display = 'none';
              scrollToBottom();
              return;
            }
            const reader = r.body.getReader();
            const decoder = new TextDecoder('utf-8');
            while(true){
              const { value, done } = await reader.read();
              if(done) break;
              const chunk = decoder.decode(value, { stream: true });
              bodyEl.innerHTML += chunk.replace(/\\n/g, '<br>');
              scrollToBottom();
            }
          } catch(e){
            bodyEl.innerHTML = '<span style=\"color:#fca5a5\"><b>Error:</b> '+ escapeHtml(e.message) +'</span>';
          } finally {
            spinner.style.display = 'none';
            scrollToBottom();
          }
        });
      })();
    ",
                             # ids
                             "send_stream",  # btn id
                             "sid",          # session_id
                             "q",            # textarea
                             backend_url_stream
    )))
  ),
  
  footer = tags$div(
    class="text-center text-muted py-3",
    tags$small(HTML(
      '&copy; 2025 · <span class="bi bi-snow"></span> Gastro · Built with <a href="https://rstudio.github.io/bslib/" target="_blank">Cholab</a>'
    ))
  )
)

# ----------------- SERVER -----------------
server <- function(input, output, session){
  hist <- reactiveVal(character())
  
  observeEvent(input$send, {
    req(nchar(input$q) > 0)
    question <- input$q
    sid <- if (nchar(input$sid) > 0) input$sid else "default"
    
    hist(c(
      hist(),
      sprintf('<div class="bubble me"><b>You</b><div class="meta">non-stream</div><div>%s</div></div>',
              htmltools::htmlEscape(question))
    ))
    updateTextAreaInput(session, "q", value = "")
    
    session$sendCustomMessage("spinner", TRUE)
    res <- safe_post_fixed(
      backend_url,
      list(question = question, session_id = sid),
      timeout_sec = 60
    )
    session$sendCustomMessage("spinner", FALSE)
    
    if(!res$ok){
      hist(c(
        hist(),
        sprintf('<div class="bubble bot"><b>Assistant</b><div class="meta">error</div><div><span style="color:#fca5a5"><b>Error:</b> %s</span></div></div>',
                htmltools::htmlEscape(res$msg))
      ))
    } else {
      ans <- res$data$answer %||% ""
      hist(c(
        hist(),
        sprintf('<div class="bubble bot"><b>Assistant</b><div class="meta">non-stream</div><div>%s</div></div>',
                htmltools::htmlEscape(ans))
      ))
    }
  })
  
  observeEvent(input$clear_chat, {
    sid <- if (nchar(input$sid) > 0) input$sid else "default"
    
    # 1) Shiny 쪽 채팅 로그 지움
    hist(character())
    # 2) 스트리밍 영역 DOM 비우기 (JS 호출)
    session$sendCustomMessage("clearStreamDOM", TRUE)
    
    # 3) 백엔드 메모리 리셋 호출 (선택)
    req <- httr2::request(backend_url_reset) |>
      httr2::req_method("POST") |>
      httr2::req_body_json(list(question = "", session_id = sid)) |>
      httr2::req_timeout(30)
    
    tryCatch({
      invisible(httr2::req_perform(req))
    }, error = function(e){
      # 리셋 실패해도 앱 동작엔 지장 없으니 로그만 남겨도 됨
      message("reset_memory error: ", conditionMessage(e))
    })
  })
  
  
  output$log <- renderUI(HTML(paste(hist(), collapse = "")))
  
  # 스크롤/스피너 핸들러
  observe({
    session$sendCustomMessage("scrollBottom", TRUE)
  })
  session$onFlushed(function(){
    session$sendCustomMessage("bindSpinner", TRUE)
  })
}

# JS 핸들러
js_handlers <- "
Shiny.addCustomMessageHandler('scrollBottom', function(x){
  const box = document.querySelector('.chat-box');
  if(box) box.scrollTop = box.scrollHeight;
});
Shiny.addCustomMessageHandler('bindSpinner', function(x){
  if(window._spinnerBound) return;
  window._spinnerBound = true;
  Shiny.addCustomMessageHandler('spinner', function(on){
    const sp = document.getElementById('loadingSpinner');
    if(sp) sp.style.display = (on ? 'inline-block' : 'none');
  });
});
Shiny.addCustomMessageHandler('clearStreamDOM', function(x){
  const box = document.getElementById('log_stream');
  if(box) box.innerHTML = '';
});
"


# bootstrap-icons 사용(아이콘)
icons_cdn <- tags$link(
  rel="stylesheet",
  href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css"
)

ui <- tagList(icons_cdn, ui_core, tags$script(HTML(js_handlers)))

# ----------------- RUN -----------------
shinyApp(ui, server)

