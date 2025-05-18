document.addEventListener("DOMContentLoaded", function () {
  // DOM Elements
  const uploadForm = document.getElementById("upload-form");
  const pdfFileInput = document.getElementById("pdf-file");
  const reindexBtn = document.getElementById("reindex-btn");
  const modelSelect = document.getElementById("model-select");
  const refreshModelsBtn = document.getElementById("refresh-models-btn");
  const vectorStoreSelect = document.getElementById("vector-store-select");
  const switchVectorStoreBtn = document.getElementById(
    "switch-vector-store-btn"
  );
  const systemToggle = document.getElementById("system-toggle");
  const systemToggleLabel = document.getElementById("system-toggle-label");
  const pdfControls = document.getElementById("pdf-controls");
  const webControls = document.getElementById("web-controls");
  const vectorStoreControls = document.getElementById("vector-store-controls");
  const documentsHeading = document.getElementById("documents-heading");
  const queryForm = document.getElementById("query-form");
  const queryInput = document.getElementById("query-input");
  const chatMessages = document.getElementById("chat-messages");
  const documentsList = document.getElementById("documents-list");
  const loadingOverlay = document.getElementById("loading-overlay");
  const toast = document.getElementById("toast");
  const toastTitle = document.getElementById("toast-title");
  const toastMessage = document.getElementById("toast-message");

  // Track current active system
  let activeSystem = systemToggle.checked ? "web" : "pdf";

  // Bootstrap toast instance
  const toastInstance = new bootstrap.Toast(toast);

  // Show notification
  function showNotification(title, message, isError = false) {
    toastTitle.textContent = title;
    toastMessage.textContent = message;
    toast.classList.remove("bg-danger", "text-white");

    if (isError) {
      toast.classList.add("bg-danger", "text-white");
    }

    toastInstance.show();
  }

  // Show loading overlay
  function showLoading() {
    loadingOverlay.classList.remove("d-none");
  }

  // Hide loading overlay
  function hideLoading() {
    loadingOverlay.classList.add("d-none");
  }

  // Add message to chat
  function addMessage(content, isUser = false) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add(
      "message",
      isUser ? "user-message" : "assistant-message"
    );

    const contentDiv = document.createElement("div");
    contentDiv.classList.add("message-content");

    // Use marked.js to render markdown if it's an assistant message
    if (!isUser) {
      contentDiv.innerHTML = marked.parse(content);
    } else {
      contentDiv.textContent = content;
    }

    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // Display retrieved documents based on active system
  function displayDocuments(documents) {
    documentsList.innerHTML = "";

    if (documents.length === 0) {
      documentsList.innerHTML = "<p>No relevant documents found.</p>";
      return;
    }

    documents.forEach((doc) => {
      const docCard = document.createElement("div");
      docCard.classList.add("card", "document-card");

      const cardHeader = document.createElement("div");
      cardHeader.classList.add("card-header");

      if (activeSystem === "pdf") {
        // PDF RAG document format
        cardHeader.textContent = `Document ${doc.index} (Source: ${doc.source})`;

        const cardBody = document.createElement("div");
        cardBody.classList.add("card-body");

        const scoreSpan = document.createElement("span");
        scoreSpan.classList.add("document-score");
        scoreSpan.textContent = `Score: ${doc.score.toFixed(4)}`;

        const contentP = document.createElement("p");
        contentP.classList.add("mb-0", "mt-2");
        contentP.textContent = doc.content;

        cardBody.appendChild(scoreSpan);
        cardBody.appendChild(contentP);

        docCard.appendChild(cardHeader);
        docCard.appendChild(cardBody);
      } else {
        // Web RAG document format
        cardHeader.textContent = doc.title || `Result ${doc.index}`;

        const cardBody = document.createElement("div");
        cardBody.classList.add("card-body");

        if (doc.url) {
          const urlLink = document.createElement("a");
          urlLink.href = doc.url;
          urlLink.target = "_blank";
          urlLink.classList.add("document-url");
          urlLink.textContent = doc.url;
          cardBody.appendChild(urlLink);
        }

        const contentP = document.createElement("p");
        contentP.classList.add("mb-0", "mt-2");
        contentP.textContent = doc.content;

        cardBody.appendChild(contentP);

        docCard.appendChild(cardHeader);
        docCard.appendChild(cardBody);
      }

      documentsList.appendChild(docCard);
    });
  }

  // Handle system toggle
  systemToggle.addEventListener("change", function () {
    activeSystem = this.checked ? "web" : "pdf";
    systemToggleLabel.textContent =
      activeSystem === "pdf" ? "PDF RAG" : "Web RAG";

    // Update UI based on active system
    if (activeSystem === "pdf") {
      pdfControls.classList.remove("d-none");
      webControls.classList.add("d-none");
      vectorStoreControls.classList.remove("d-none");
      queryInput.placeholder = "Ask a question about your PDFs...";
      documentsHeading.textContent = "Retrieved Documents";
    } else {
      pdfControls.classList.add("d-none");
      webControls.classList.remove("d-none");
      vectorStoreControls.classList.add("d-none");
      queryInput.placeholder = "Ask anything to search the web...";
      documentsHeading.textContent = "Search Results";
    }

    // Clear previous results
    documentsList.innerHTML = "";

    // Send request to server to switch system
    showLoading();

    fetch("/switch-system", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        system: activeSystem,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.error) {
          showNotification("System Switch Error", data.error, true);
        } else {
          showNotification(
            "Success",
            `Switched to ${activeSystem.toUpperCase()} RAG system.`
          );
        }
      })
      .catch((error) => {
        showNotification(
          "System Switch Error",
          "An error occurred while switching systems.",
          true
        );
        console.error("System switch error:", error);
      })
      .finally(() => {
        hideLoading();
      });
  });

  // Handle file upload
  uploadForm.addEventListener("submit", function (e) {
    e.preventDefault();

    const file = pdfFileInput.files[0];
    if (!file) {
      showNotification("Error", "Please select a PDF file to upload.", true);
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    showLoading();

    fetch("/upload", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.error) {
          showNotification("Upload Error", data.error, true);
        } else {
          showNotification(
            "Success",
            `File ${data.filename} uploaded and indexed successfully.`
          );
          // Reload the page to update the PDF list
          window.location.reload();
        }
      })
      .catch((error) => {
        showNotification(
          "Upload Error",
          "An error occurred during upload.",
          true
        );
        console.error("Upload error:", error);
      })
      .finally(() => {
        hideLoading();
      });
  });

  // Handle reindexing
  reindexBtn.addEventListener("click", function () {
    showLoading();

    fetch("/index", {
      method: "POST",
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.error) {
          showNotification("Indexing Error", data.error, true);
        } else {
          showNotification("Success", "Documents reindexed successfully.");
        }
      })
      .catch((error) => {
        showNotification(
          "Indexing Error",
          "An error occurred during indexing.",
          true
        );
        console.error("Indexing error:", error);
      })
      .finally(() => {
        hideLoading();
      });
  });

  // Handle refresh models button
  refreshModelsBtn.addEventListener("click", function () {
    showLoading();

    // Fetch available models from the server
    fetch("/models")
      .then((response) => response.json())
      .then((models) => {
        // Clear the current options
        modelSelect.innerHTML = "";

        // Add the new options
        models.forEach((model) => {
          const option = document.createElement("option");
          option.value = model;
          option.textContent = model;
          modelSelect.appendChild(option);
        });

        showNotification("Success", `Found ${models.length} available models.`);
      })
      .catch((error) => {
        console.error("Error fetching models:", error);
        showNotification("Error", "Failed to fetch available models.", true);
      })
      .finally(() => {
        hideLoading();
      });
  });

  // Handle vector store switching
  switchVectorStoreBtn.addEventListener("click", function () {
    const vectorStoreType = vectorStoreSelect.value;
    showLoading();

    fetch("/switch-vector-store", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        vector_store_type: vectorStoreType,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.error) {
          showNotification("Vector Store Error", data.error, true);
        } else {
          showNotification(
            "Success",
            `Switched to ${data.vector_store_type.toUpperCase()} vector store.`
          );
        }
      })
      .catch((error) => {
        showNotification(
          "Vector Store Error",
          "An error occurred while switching vector stores.",
          true
        );
        console.error("Vector store error:", error);
      })
      .finally(() => {
        hideLoading();
      });
  });

  // Handle query submission
  queryForm.addEventListener("submit", function (e) {
    e.preventDefault();

    const question = queryInput.value.trim();
    if (!question) {
      return;
    }

    const model = modelSelect.value;

    // Add user message to chat
    addMessage(question, true);

    // Create a placeholder for the assistant's response
    const assistantPlaceholder = document.createElement("div");
    assistantPlaceholder.classList.add("message", "assistant-message");

    const contentDiv = document.createElement("div");
    contentDiv.classList.add("message-content");
    contentDiv.textContent = "Thinking...";

    assistantPlaceholder.appendChild(contentDiv);
    chatMessages.appendChild(assistantPlaceholder);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // Clear input
    queryInput.value = "";

    showLoading();

    // First, get the retrieved documents
    fetch("/query", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: question,
        model: model,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.error) {
          contentDiv.textContent = `Error: ${data.error}`;
          showNotification("Query Error", data.error, true);
        } else {
          // Display retrieved documents or web search results
          if (activeSystem === "web") {
            if (typeof displaySearchResults === "function") {
              displaySearchResults(data.documents);
            } else {
              displayDocuments(data.documents);
            }
          } else {
            displayDocuments(data.documents);
          }

          // Update active system if it changed on the server
          if (data.active_system && data.active_system !== activeSystem) {
            activeSystem = data.active_system;
            systemToggle.checked = activeSystem === "web";
            systemToggleLabel.textContent =
              activeSystem === "pdf" ? "PDF RAG" : "Web RAG";

            // Update UI based on active system
            if (activeSystem === "pdf") {
              pdfControls.classList.remove("d-none");
              webControls.classList.add("d-none");
              vectorStoreControls.classList.remove("d-none");
              queryInput.placeholder = "Ask a question about your PDFs...";
              documentsHeading.textContent = "Retrieved Documents";
            } else {
              pdfControls.classList.add("d-none");
              webControls.classList.remove("d-none");
              vectorStoreControls.classList.add("d-none");
              queryInput.placeholder = "Ask anything to search the web...";
              documentsHeading.textContent = "Search Results";
            }
          }

          // Display vector store type (only for PDF RAG)
          if (activeSystem === "pdf" && data.vector_store_type) {
            const vectorStoreType = data.vector_store_type;
            console.log(`Using vector store: ${vectorStoreType}`);

            // Update vector store select if needed
            if (
              vectorStoreType &&
              vectorStoreSelect.value !== vectorStoreType
            ) {
              vectorStoreSelect.value = vectorStoreType;
            }
          }

          // Hide loading overlay before starting streaming
          hideLoading();

          // Now start streaming the response
          const streamUrl = `/stream?question=${encodeURIComponent(
            question
          )}&model=${encodeURIComponent(model)}`;
          const eventSource = new EventSource(streamUrl);

          let responseText = "";

          console.log("Starting EventSource connection to:", streamUrl);

          eventSource.onmessage = function (event) {
            console.log("Received event:", event.data);
            if (event.data === "[DONE]") {
              eventSource.close();
            } else {
              responseText += event.data;
              contentDiv.innerHTML = marked.parse(responseText);
              chatMessages.scrollTop = chatMessages.scrollHeight;
            }
          };

          eventSource.onerror = function (error) {
            console.error("EventSource error:", error);
            eventSource.close();
            if (!responseText) {
              contentDiv.textContent = "Error: Failed to generate response.";
            }
            hideLoading();
          };

          // Add event listener for when the connection is opened
          eventSource.onopen = function () {
            console.log("EventSource connection opened");
          };
        }
      })
      .catch((error) => {
        contentDiv.textContent = "Error: Failed to process query.";
        showNotification(
          "Query Error",
          "An error occurred while processing your query.",
          true
        );
        console.error("Query error:", error);
        hideLoading();
      });
  });
});
