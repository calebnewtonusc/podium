"use client";

import { useEffect, useRef } from "react";
import Nav from "@/components/nav";
import Waitlist from "@/components/waitlist";

const ACCENT = "#F59E0B";
const HUB_URL = "https://specialized-model-startups.vercel.app";

function useScrollReveal() {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { el.classList.add("visible"); obs.unobserve(el); } },
      { threshold: 0.12 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);
  return ref;
}

function SectionLabel({ label }: { label: string }) {
  const ref = useScrollReveal();
  return (
    <div ref={ref} className="reveal flex items-center gap-5 mb-12">
      <span className="text-xs font-semibold uppercase tracking-[0.18em] text-gray-400 shrink-0">{label}</span>
      <div className="flex-1 h-px bg-gray-100" />
    </div>
  );
}

// ── Sub-components extracted so hooks are called at the top level of a component ─

function StepCard({ step, title, desc }: { step: string; title: string; desc: string }) {
  const ref = useScrollReveal();
  return (
    <div key={step} ref={ref} className="reveal-scale rounded-2xl border border-gray-100 bg-white p-8">
      <div className="text-xs font-bold uppercase tracking-widest mb-4" style={{ color: ACCENT }}>{step}</div>
      <h3 className="serif font-semibold text-lg mb-3 text-gray-900">{title}</h3>
      <p className="text-sm text-gray-500 leading-relaxed">{desc}</p>
    </div>
  );
}

function CapabilityCard({ icon, title, desc }: { icon: React.ReactNode; title: string; desc: string }) {
  const ref = useScrollReveal();
  return (
    <div
      ref={ref}
      className="reveal rounded-2xl border border-gray-100 p-7 flex gap-5 hover:border-gray-200 transition-colors"
    >
      <div
        className="shrink-0 w-10 h-10 rounded-xl flex items-center justify-center"
        style={{ backgroundColor: `${ACCENT}10` }}
      >
        {icon}
      </div>
      <div>
        <h3 className="font-semibold text-sm text-gray-900 mb-1.5">{title}</h3>
        <p className="text-sm text-gray-500 leading-relaxed">{desc}</p>
      </div>
    </div>
  );
}

function StatCard({ stat, label, sub }: { stat: string; label: string; sub: string }) {
  const ref = useScrollReveal();
  return (
    <div
      ref={ref}
      className="reveal rounded-2xl border p-8"
      style={{ borderColor: `${ACCENT}20` }}
    >
      <div className="text-3xl font-bold tracking-tight mb-2" style={{ color: ACCENT }}>{stat}</div>
      <div className="text-sm font-semibold text-gray-800 mb-1">{label}</div>
      <div className="text-xs text-gray-400">{sub}</div>
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function Home() {
  return (
    <div className="min-h-screen bg-white text-[#0a0a0a] overflow-x-hidden">
      <Nav />

      {/* Hero */}
      <section className="relative min-h-screen flex flex-col justify-center px-6 pt-14 overflow-hidden">
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            backgroundImage: `radial-gradient(circle at 20% 30%, ${ACCENT}08 0%, transparent 50%), radial-gradient(circle at 80% 70%, ${ACCENT}06 0%, transparent 50%)`,
          }}
        />

        <div className="relative max-w-5xl mx-auto w-full py-20">
          <div className="fade-up delay-0 mb-8">
            <span
              className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold border"
              style={{ color: ACCENT, borderColor: `${ACCENT}30`, backgroundColor: `${ACCENT}08` }}
            >
              <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: ACCENT }} />
              Training &middot; 18&times; A6000 &middot; ETA Q3 2026
            </span>
          </div>

          <h1 className="fade-up delay-1 text-[clamp(3.5rem,10vw,7rem)] font-bold leading-[0.92] tracking-tight mb-6">
            <span className="serif font-light italic" style={{ color: ACCENT }}>Po</span>
            <span>dium</span>
          </h1>

          <p className="fade-up delay-2 serif text-[clamp(1.25rem,3vw,2rem)] font-light text-gray-500 mb-4 max-w-xl">
            Competition-grade ML engineering, automated.
          </p>

          <p className="fade-up delay-3 text-sm text-gray-400 leading-relaxed max-w-lg mb-10">
            The first model to use cross-validation score as a verifiable RL reward&nbsp;— the same way DeepSeek-R1 used math correctness, but for Kaggle.
          </p>

          <div className="fade-up delay-4">
            <Waitlist />
          </div>
        </div>
      </section>

      {/* The Problem */}
      <section className="px-6 py-24 max-w-5xl mx-auto">
        <SectionLabel label="The Problem" />
        <div className="grid md:grid-cols-2 gap-6">
          <div className="reveal rounded-2xl border border-gray-100 p-8 bg-gray-50/50">
            <p className="text-xs font-semibold uppercase tracking-widest text-gray-400 mb-5">What general models do</p>
            <ul className="space-y-3 text-sm text-gray-500 leading-relaxed">
              <li className="flex gap-3">
                <span className="text-gray-300 mt-0.5">&#8212;</span>
                Scaffolded agents achieve only 17% Kaggle medal rate
              </li>
              <li className="flex gap-3">
                <span className="text-gray-300 mt-0.5">&#8212;</span>
                Pattern-match on syntax, not on why solutions win
              </li>
              <li className="flex gap-3">
                <span className="text-gray-300 mt-0.5">&#8212;</span>
                Generate boilerplate notebooks with no reasoning
              </li>
              <li className="flex gap-3">
                <span className="text-gray-300 mt-0.5">&#8212;</span>
                Can&apos;t evaluate quality without running CV — so they don&apos;t
              </li>
            </ul>
          </div>

          <div
            className="reveal rounded-2xl border p-8"
            style={{ borderColor: `${ACCENT}25`, backgroundColor: `${ACCENT}05` }}
          >
            <p className="text-xs font-semibold uppercase tracking-widest mb-5" style={{ color: ACCENT }}>What Podium does</p>
            <ul className="space-y-3 text-sm leading-relaxed text-gray-700">
              <li className="flex gap-3">
                <svg className="mt-0.5 shrink-0" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                Bakes grandmaster reasoning into weights — the WHY, not the code
              </li>
              <li className="flex gap-3">
                <svg className="mt-0.5 shrink-0" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                Uses CV score improvement as a verifiable, automatic reward
              </li>
              <li className="flex gap-3">
                <svg className="mt-0.5 shrink-0" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                Trained on 280k notebooks + 50k winning competition writeups
              </li>
              <li className="flex gap-3">
                <svg className="mt-0.5 shrink-0" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                Measures what matters: leaderboard score delta, not test coverage
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="px-6 py-24 bg-gray-50/50">
        <div className="max-w-5xl mx-auto">
          <SectionLabel label="How it works" />
          <div className="grid md:grid-cols-3 gap-6">
            <StepCard
              step="01"
              title="Supervised Fine-Tuning"
              desc="800k (task, solution, cv_score) triples from 280k Kaggle notebooks and 50k gold-medal writeups. Base: Qwen2.5-7B-Coder-Instruct. Podium learns the mapping from competition brief to winning approach."
            />
            <StepCard
              step="02"
              title="RL with Verifiable Reward"
              desc="Cross-validation score improvement is the reward — measurable and automatic, requiring no human labels. The same insight that made DeepSeek-R1 work, applied to ML competition engineering."
            />
            <StepCard
              step="03"
              title="DPO Alignment"
              desc="Direct Preference Optimization on (higher-CV, lower-CV) pairs. Podium learns to prefer ensembling over single models, feature engineering over raw features, and proper temporal CV over naive splits."
            />
          </div>
        </div>
      </section>

      {/* Capabilities */}
      <section className="px-6 py-24 max-w-5xl mx-auto">
        <SectionLabel label="Capabilities" />
        <div className="grid sm:grid-cols-2 gap-5">
          <CapabilityCard
            icon={
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/>
              </svg>
            }
            title="Tabular ML at scale"
            desc="Gradient boosting ensembles (XGBoost + LightGBM + CatBoost), automated feature engineering, target encoding, and stacking — Podium knows when each technique wins."
          />
          <CapabilityCard
            icon={
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>
              </svg>
            }
            title="Computer Vision"
            desc="ViT fine-tuning, augmentation strategies from top CV notebooks, test-time augmentation, and multi-scale inference — tuned for competition leaderboard, not just accuracy metrics."
          />
          <CapabilityCard
            icon={
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
              </svg>
            }
            title="NLP competition engineering"
            desc="DeBERTa and RoBERTa stacking, custom pooling strategies, mean/max/CLS pooling ensembles, and pseudo-labeling pipelines — the full grandmaster NLP toolkit."
          />
          <CapabilityCard
            icon={
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
              </svg>
            }
            title="Time series strategy"
            desc="LSTM ensembles with temporal CV, walk-forward validation, fourier features, and lag engineering — Podium understands data leakage in time series and avoids it by default."
          />
        </div>
      </section>

      {/* The numbers */}
      <section className="px-6 py-24 bg-gray-50/50">
        <div className="max-w-5xl mx-auto">
          <SectionLabel label="The numbers" />
          <div className="grid sm:grid-cols-3 gap-6">
            <StatCard stat="800k" label="Training pairs" sub="280k notebooks + 50k writeups" />
            <StatCard stat="Qwen2.5-7B" label="Base model" sub="Coder-Instruct" />
            <StatCard stat="CV Score" label="Reward signal" sub="Cross-validation improvement" />
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="px-6 py-12 border-t border-gray-100">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-gray-400">
          <p>
            Part of the{" "}
            <a href={HUB_URL} className="underline underline-offset-2 hover:text-gray-600 transition-colors">
              Specialist AI
            </a>{" "}
            portfolio by{" "}
            <a
              href="https://github.com/calebnewtonusc"
              target="_blank"
              rel="noopener noreferrer"
              className="underline underline-offset-2 hover:text-gray-600 transition-colors"
            >
              Caleb Newton &middot; calebnewtonusc
            </a>{" "}
            &middot; 2026
          </p>
        </div>
      </footer>
    </div>
  );
}
