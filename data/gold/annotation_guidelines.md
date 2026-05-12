# Annotation Guidelines

Evaluation unit:
One action-unit/object pair.

Gold roles:

1. Target / Affected object
Object that is inspected, checked, reviewed, approved, rejected, changed, or otherwise directly acted upon.

2. Result / Output object
Object that is created, generated, issued, prepared, produced, registered, or formed as a result of the action.

3. Source / Input object
Object used as source, basis, input, origin, or reference for another action, decision, or output.

4. Transferred object
Object sent, submitted, forwarded, transmitted, delivered, returned, or moved to another participant, system, or process context.

5. Support / Instrument object
Tool, channel, medium, form, system, or resource used to perform the action, but not the main affected object.

6. Recipient-linked element
Participant, system, department, role, or actor that receives or is the destination/beneficiary of a transferred object.

Rules:
- PET labels are not final thesis roles.
- PET `uses` only means that an activity is connected to activity data.
- PET `actor recipient` is strong evidence for Recipient-linked element.
- If the case is unclear, mark gold_status = ambiguous and do not include it in final evaluation.
- If the object span is wrong or unusable, mark gold_status = excluded.