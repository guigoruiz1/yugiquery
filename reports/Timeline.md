---
layout: base
title: Timeline
permalink: /reports/Timeline/
---

<style>
	.report-frame-shell {
		/* Break out of narrow content columns in page layouts */
		position: relative;
		left: 50%;
		right: 50%;
		margin-left: -50vw;
		margin-right: -50vw;
		width: 100vw;
		display: flex;
		justify-content: center;
	}

	.report-frame {
		/* Keep it roomy on desktop but still responsive on smaller screens */
		width: min(96vw, 1600px);
		min-width: 50vw;
		height: 100vh;
		border: 0;
		overflow: hidden;
		display: block;
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
		src="../Timeline.html"
		title="Timeline report"
		loading="lazy"
		scrolling="no"
	></iframe>
</div>

<script>
	(function () {
		const frame = document.getElementById("report-frame");
		if (!frame) return;

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
