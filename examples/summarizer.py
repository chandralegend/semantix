from semantix import Semantic, with_llm

llm = None  # Replace with the LLM model


@with_llm("Summarize the Given Text", llm)
def summarize(text: Semantic[str, "Text to Summarize"]) -> Semantic[str, "Summary"]:  # type: ignore
    ...


text = """
Nuclear power is the use of nuclear reactions to produce electricity. Nuclear power can be obtained from nuclear 
fission, nuclear decay and nuclear fusion reactions. Presently, the vast majority of electricity from nuclear power
is produced by nuclear fission of uranium and plutonium in nuclear power plants. Nuclear decay processes are used
in niche applications such as radioisotope thermoelectric generators in some space probes such as Voyager 2. 
Generating electricity from fusion power remains the focus of international research.

Most nuclear power plants use thermal reactors with enriched uranium in a once-through fuel cycle. Fuel is removed
when the percentage of neutron absorbing atoms becomes so large that a chain reaction can no longer be sustained,
typically three years. It is then cooled for several years in on-site spent fuel pools before being transferred to
long-term storage. The spent fuel, though low in volume, is high-level radioactive waste. While its radioactivity
decreases exponentially, it must be isolated from the biosphere for hundreds of thousands of years, though newer
technologies (like fast reactors) have the potential to significantly reduce this. Because the spent fuel is still
mostly fissionable material, some countries (e.g. France and Russia) reprocess their spent fuel by extracting fissile
and fertile elements for fabrication in new fuel, although this process is more expensive than producing new fuel from
mined uranium. All reactors breed some plutonium-239, which is found in the spent fuel, and because Pu-239 is the
preferred material for nuclear weapons, reprocessing is seen as a weapon proliferation risk.
"""

summarize(text)
