* architecture.md should be a living document describing high-level interfaces and their interactions
* doc strings can go into more of the implementation detail
* Reason carefully about what should be long-term interface guarantees and what is a result of the current implementation
* Allowing python to throw AttributeErrors and KeyErrors when interfaces are misused is better than a lot of checking for programmer error. It's definitely better than turning invalid input into continues/hidden ignored conditions
* Producing better errors for things in the incoming document is valuable
* ~/ai/vibevoice/.venv is the venv to use
* Commit after changes
