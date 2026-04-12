---
layout: base
title: @title@
permalink: /reports/@title@/
---

<style>
	html,
	body {
		overflow-x: clip;
	}

	.report-frame-shell {
		/* Break out of narrow content columns in page layouts */
		position: relative;
		margin-left: calc(50% - 50vw);
		margin-right: calc(50% - 50vw);
		width: 100vw;
		max-width: 100vw;
		display: flex;
		justify-content: center;
		overflow-x: clip;
	}

	.report-frame {
		/* Keep it roomy on desktop but still responsive on smaller screens */
		position: relative;
		width: min(96vw, 1600px);
		min-width: 50vw;
		height: 100vh;
		border: 0;
		overflow: hidden;
		display: block;
	}

	/* Keep common nav overlays above embedded reports */
	header,
	.site-header,
	.navbar,
	.navbar-menu,
	.dropdown-menu,
    .site-nav,
	.menu {
		position: relative;
		z-index: 10;
	}

	@media (max-width: 900px) {
		.report-frame {
			/* Avoid overflow on mobile while keeping a generous width */
			min-width: 92vw;
			width: 92vw;
		}
	}

</style>

<div class="report-frame-shell">
	<iframe
		id="report-frame"
		class="report-frame"
		src="../@title@.html"
		title="@title@ report"
		loading="lazy"
		scrolling="no"
	></iframe>
</div>

<script>
	(function () {
		const frame = document.getElementById("report-frame");
		if (!frame) return;

		const injectCenteringCSS = () => {
			try {
				const doc = frame.contentDocument;
				if (!doc) return;

				// Create and inject centering styles into iframe
				const style = doc.createElement("style");
				style.textContent = `
					.jp-RenderedSVG {
						display: flex !important;
						justify-content: center !important;
						align-items: center !important;
					}

					/* Remove notebook panel background tint from exported pages */
					.jp-Notebook,
					.jp-notebook,
					.jp-NotebookPanel-notebook {
						background: transparent !important;
					}
				`;
				doc.head.appendChild(style);
			} catch (_) {
				// Ignore if injection fails.
			}
		};

		const setFrameHeight = () => {
			try {
				const doc = frame.contentDocument;
				if (!doc) return;
				const body = doc.body;
				const html = doc.documentElement;
				if (!body || !html) return;

				const height = Math.max(
					body.scrollHeight,
					body.offsetHeight,
					html.clientHeight,
					html.scrollHeight,
					html.offsetHeight
				);

				frame.style.height = height + "px";
			} catch (_) {
				// Ignore cross-origin or transient access errors.
			}
		};

		frame.addEventListener("load", () => {
			injectCenteringCSS();
			setFrameHeight();
			setTimeout(setFrameHeight, 200);
			setTimeout(setFrameHeight, 1000);

			try {
				const doc = frame.contentDocument;
				if (!doc) return;
				const observer = new ResizeObserver(setFrameHeight);
				observer.observe(doc.documentElement);
				observer.observe(doc.body);
			} catch (_) {
				// Ignore if observer cannot be attached.
			}
		});

		window.addEventListener("resize", setFrameHeight);
	})();
</script>
